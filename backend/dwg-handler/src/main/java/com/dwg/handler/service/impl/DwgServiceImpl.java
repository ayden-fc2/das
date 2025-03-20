package com.dwg.handler.service.impl;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.dao.*;
import com.dwg.handler.entity.*;
import com.dwg.handler.service.DwgService;
import com.dwg.handler.utils.GraphProcessor;
import com.dwg.handler.utils.JsonProcessor;
import com.example.common.exception.MyException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.bootstrap.encrypt.KeyProperties;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

@Service
public class DwgServiceImpl implements DwgService {

    @Value("${custom.save-file-path-dwg}" + "/")
    private String dwgUploadFilePath;

    @Value("${custom.back-file-path-dwg}" + "/")
    private String dwgReturnUrl;

    @Autowired
    GraphProcessor graphProcessor;

    @Autowired
    UploadDwgStMapper uploadDwgStMapper;

    @Autowired
    BlockStMapper blockStMapper;

    @Autowired
    InsertStMapper insertStMapper;

    @Autowired
    JsonProcessor jsonProcessor;

    @Autowired
    KeyPipeStMapper keyPipeStMapper;

    @Autowired
    VirtualNodeStMapper virtualNodeStMapper;
    @Autowired
    private KeyProperties keyProperties;

    // 使用libreDWG生成JSON文件
    @Override
    public Boolean genAnalysis(int userId, String projectName, String dwgPath, int isPublic) {
        String file;
        if (dwgPath.startsWith(dwgReturnUrl)) {
            file = dwgPath.substring(dwgReturnUrl.length());
        } else {
            throw new MyException("The dwgPath does not start with dwgReturnUrl");
        }
        String fileName = file.split("\\.")[0];
        String extension = file.split("\\.")[1];
        if (!extension.equals("dwg")) {
            throw new MyException("The extension is not dwg");
        }
        String fullPath = dwgUploadFilePath + file;

        // 需要执行的命令
        String command = "dwgread " + fullPath + " -O JSON -o " + dwgUploadFilePath + fileName + ".json";
        System.out.println(command);
        // 等待命令执行完毕
        try {
            // 使用 Runtime 执行命令
            Process process = Runtime.getRuntime().exec(command);
            // 获取命令的输出
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
            int exitCode = process.waitFor();
            System.out.println("Command executed with exit code: " + exitCode);
            String unhandledJson = dwgUploadFilePath + fileName + ".json";
            String handledJson = dwgUploadFilePath + fileName + "_handled" + ".json";
            jsonProcessor.processJsonFile(unhandledJson, handledJson);
            String jsonPath = dwgReturnUrl + fileName + "_handled" + ".json";
            // 新增一条数据
            UploadDwgSt newDwgAna = new UploadDwgSt();
            newDwgAna.setProjectName(projectName);
            newDwgAna.setDwgPath(dwgPath);
            newDwgAna.setIsPublic(isPublic);
            newDwgAna.setJsonPath(jsonPath);
            newDwgAna.setUserId(userId);
            return uploadDwgStMapper.insert(newDwgAna);
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException(e.getMessage());
        }
    }

    @Override
    public boolean isAnalysed(long projectId) {
        return uploadDwgStMapper.isAnalysed(projectId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean genGrahpML(int userId, JSONArray blockData, JSONArray insertsData, JSONArray pipesData, Long projectId) {
        try {
            List<InsertSt> keyNodes = new ArrayList<>();
            // 入库 blockData
            blockStMapper.deleteByDwgId(projectId);
            insertStMapper.deleteByDwgId(projectId);
            keyPipeStMapper.deleteByDwgId(projectId);
            virtualNodeStMapper.deleteByDwgId(projectId);
            // TODO: 通过LLM获取节点类型
            for (int i = 0; i < blockData.size(); i++) {
                JSONObject block = blockData.getJSONObject(i);
                BlockSt blockSt = new BlockSt();
                blockSt.setBlockId(-1);
                blockSt.setDwgId(projectId);
                blockSt.setBlockCount((int)block.get("count"));
                blockSt.setBlockName((String)block.get("name"));
                blockSt.setHandle0(block.getLong("handle0"));
                blockSt.setHandle1(block.getLong("handle1"));
                blockSt.setSvg(block.getString("svgString"));
                blockStMapper.insertBlockSt(blockSt);
            }
            for (int i = 0; i < insertsData.size(); i++) {
                JSONObject insert = insertsData.getJSONObject(i);
                InsertSt insertSt = new InsertSt();
                insertSt.setDwgId(projectId);
                insertSt.setInsertId(-1);
                insertSt.setBlockHandle0(insert.getLong("blockHandle0"));
                insertSt.setBlockHandle1(insert.getLong("blockHandle1"));
                insertSt.setInsertHandle0(insert.getLong("handle0"));
                insertSt.setInsertHandle1(insert.getLong("handle1"));
                insertSt.setBoxWidth(insert.getDouble("boxWidth"));
                insertSt.setBoxHeight(insert.getDouble("boxHeight"));
                insertSt.setCenterPtX(insert.getJSONArray("centerPt").getDouble(0));
                insertSt.setCenterPtY(insert.getJSONArray("centerPt").getDouble(1));
                keyNodes.add(insertSt);
            }
            // 图算法处理pipesData，得到graph结构，以及insert间的上下游关系
            // Step 0: 获取全部候选边
            List<KeyPipeSt> pipes = new ArrayList<>();
            List<KeyPipeSt> candidatePipes = new ArrayList<>();
            List<VirtualNodeSt> virtualNodeStList = new ArrayList<>();
            for (int i = 0; i < pipesData.size(); i++) {
                JSONObject pipe = pipesData.getJSONObject(i);
                KeyPipeSt keyPipeSt = new KeyPipeSt();
                keyPipeSt.setKeyPipeId(-1);
                keyPipeSt.setStartX(pipe.getDouble("startX"));
                keyPipeSt.setStartY(pipe.getDouble("startY"));
                keyPipeSt.setEndX(pipe.getDouble("endX"));
                keyPipeSt.setEndY(pipe.getDouble("endY"));
                keyPipeSt.setDwgId(projectId);
                pipes.add(keyPipeSt);
            }
            // Step 1: 首次遍历keyNodes，得到第一层虚拟节点
            for (InsertSt keyNode : keyNodes) {
                for (KeyPipeSt pipe : pipes) {
                    int result = graphProcessor.inNodeBox(keyNode, pipe);
                    if (result == 0) {
                        // 起点在关键节点内
                        // 入库虚拟节点
                        VirtualNodeSt v = new VirtualNodeSt();
                        v.setDwgId(projectId);
                        v.setVNodeId(-1);
                        v.setUuid(UUID.randomUUID().toString());
                        v.setX(pipe.getEndX());
                        v.setY(pipe.getEndY());
                        String vUUID = graphProcessor.hasVNode(v, virtualNodeStList);
                        if (vUUID == null || vUUID.isEmpty()) {
                            virtualNodeStList.add(v);
                            vUUID = v.getUuid();
                        }
                        // 入库边
                        KeyPipeSt keyPipe = new KeyPipeSt();
                        keyPipe.setKeyPipeId(-1);
                        keyPipe.setStartX(pipe.getStartX());
                        keyPipe.setStartY(pipe.getStartY());
                        keyPipe.setEndX(pipe.getEndX());
                        keyPipe.setEndY(pipe.getEndY());
                        keyPipe.setDwgId(projectId);
                        keyPipe.setStartKeyHandle0(keyNode.getInsertHandle0());
                        keyPipe.setStartKeyHandle1(keyNode.getInsertHandle1());
                        keyPipe.setVEndUUID(vUUID);
                        candidatePipes.add(keyPipe);
                    } else if (result == 1) {
                        // 终点在关键节点内
                        // 入库虚拟节点
                        VirtualNodeSt v1 = new VirtualNodeSt();
                        v1.setDwgId(projectId);
                        v1.setVNodeId(-1);
                        v1.setUuid(UUID.randomUUID().toString());
                        v1.setX(pipe.getStartX());
                        v1.setY(pipe.getStartY());
                        String v1UUID = graphProcessor.hasVNode(v1, virtualNodeStList);
                        if (v1UUID == null || v1UUID.isEmpty()) {
                            virtualNodeStList.add(v1);
                            v1UUID = v1.getUuid();
                        }
                        // 入库边
                        KeyPipeSt keyPipe1 = new KeyPipeSt();
                        keyPipe1.setKeyPipeId(-1);
                        keyPipe1.setStartX(pipe.getStartX());
                        keyPipe1.setStartY(pipe.getStartY());
                        keyPipe1.setEndX(pipe.getEndX());
                        keyPipe1.setEndY(pipe.getEndY());
                        keyPipe1.setDwgId(projectId);
                        keyPipe1.setEndKeyHandle0(keyNode.getInsertHandle0());
                        keyPipe1.setEndKeyHandle1(keyNode.getInsertHandle1());
                        keyPipe1.setVStartUUID(v1UUID);
                        candidatePipes.add(keyPipe1);
                    }
                }
            }

            // Step 2: 迭代遍历vNodes
            List<VirtualNodeSt> addListV = new ArrayList<>();
            List<VirtualNodeSt> deleteListV = new ArrayList<>();
            while (graphProcessor.checkVirtualNodes(virtualNodeStList)) {
                addListV.clear();
                // 先处理所有的虚拟节点，如果在关键节点内，则更新相关管道，删除虚拟节点，最后对管道去重
                for (VirtualNodeSt v : virtualNodeStList) {
                    if (v.getFinished() == 0) {
                        for (InsertSt keyNode : keyNodes) {
                            if (graphProcessor.ponitInNodeBox(keyNode, v)) {
                                graphProcessor.updateV2Key(v, keyNode, candidatePipes, virtualNodeStList);
                                deleteListV.add(v);
                            }
                        }
                    }
                }
                virtualNodeStList.removeAll(deleteListV);
                candidatePipes = graphProcessor.uniquePipes(candidatePipes);
                // 对剩下的虚拟节点去寻找候选边，如果在候选边的端点上，则分情况讨论
                for (VirtualNodeSt v : virtualNodeStList) {
                    if (v.getFinished() == 0) {
                        for (KeyPipeSt pipe : pipes) {
                            int ifVNodeInPipe = graphProcessor.vNodeInLine(v, pipe);
                            if (ifVNodeInPipe >= 0) {
                                VirtualNodeSt v1 = graphProcessor.getAnotherVNode(pipe, ifVNodeInPipe, virtualNodeStList);
                                if (v1 != null) {
                                    // 如果候选边的另一个端点是已有的虚拟节点，则处理管道后添加进管道列表，去重
                                    KeyPipeSt newPipe = new KeyPipeSt();
                                    newPipe.setDwgId(projectId);
                                    newPipe.setKeyPipeId(-1);
                                    newPipe.setStartX(pipe.getStartX());
                                    newPipe.setStartY(pipe.getStartY());
                                    newPipe.setEndX(pipe.getEndX());
                                    newPipe.setEndY(pipe.getEndY());
                                    newPipe.setVStartUUID(ifVNodeInPipe == 0 ? v.getUuid() : v1.getUuid());
                                    newPipe.setVEndUUID(ifVNodeInPipe == 1 ? v.getUuid() : v1.getUuid());
                                    candidatePipes.add(newPipe);
                                } else if (graphProcessor.isAnotherNodeDeleted(pipe, ifVNodeInPipe, deleteListV)) {
                                    // 如果另一个端点是已删除的虚拟节点(key节点)，则不处理管道
                                } else {
                                    // 如果候选边的另一个端点是未知节点，则创建新的虚拟节点以及新的管道
                                    VirtualNodeSt newV = new VirtualNodeSt();
                                    newV.setDwgId(projectId);
                                    newV.setUuid(UUID.randomUUID().toString());
                                    newV.setX(ifVNodeInPipe == 0 ? pipe.getEndX() : pipe.getStartX());
                                    newV.setY(ifVNodeInPipe == 0 ? pipe.getEndY() : pipe.getStartY());
                                    newV.setDwgId(projectId);
                                    newV.setVNodeId(-1);
                                    addListV.add(newV);

                                    KeyPipeSt newPipe = new KeyPipeSt();
                                    newPipe.setDwgId(projectId);
                                    newPipe.setKeyPipeId(-1);
                                    newPipe.setStartX(pipe.getStartX());
                                    newPipe.setStartY(pipe.getStartY());
                                    newPipe.setEndX(pipe.getEndX());
                                    newPipe.setEndY(pipe.getEndY());
                                    newPipe.setVStartUUID(ifVNodeInPipe == 0 ? v.getUuid() : newV.getUuid());
                                    newPipe.setVEndUUID(ifVNodeInPipe == 1 ? v.getUuid() : newV.getUuid());
                                    candidatePipes.add(newPipe);
                                }
                            }
                        }
                    }
                }
                candidatePipes = graphProcessor.uniquePipes(candidatePipes);

                // 对剩下的虚拟节点，查看它是否在候选边的路径上，如果在，则分情况讨论
                ListIterator<VirtualNodeSt> iterator = virtualNodeStList.listIterator();
                while (iterator.hasNext()) {
                    VirtualNodeSt v = iterator.next();
                    if (v.getFinished() == 0) {
                        ListIterator<KeyPipeSt> iterator2 = pipes.listIterator();
                        while (iterator2.hasNext()) {
                            KeyPipeSt pipe = iterator2.next();
                            if (graphProcessor.vNodeOnLineRoad(v, pipe)) {
                                KeyPipeSt keyPipe = graphProcessor.isPipeListHasPipe(candidatePipes, pipe);
                                if (keyPipe != null) {
                                    // 如果该候选边已经在管道库里，继承该管道的属性，拆成两个管道更新管道库，对节点进行进一步处理
                                    KeyPipeSt newCandidatePipe1 = keyPipe.clone();
                                    KeyPipeSt newCandidatePipe2 = keyPipe.clone();
                                    newCandidatePipe1.setVEndUUID(v.getUuid());
                                    newCandidatePipe1.setEndKeyHandle0(-1);
                                    newCandidatePipe1.setEndKeyHandle1(-1);
                                    newCandidatePipe1.setEndX(v.getX());
                                    newCandidatePipe1.setEndY(v.getY());

                                    newCandidatePipe2.setVStartUUID(v.getUuid());
                                    newCandidatePipe2.setStartKeyHandle0(-1);
                                    newCandidatePipe2.setStartKeyHandle1(-1);
                                    newCandidatePipe2.setStartX(v.getX());
                                    newCandidatePipe2.setStartY(v.getY());

                                    candidatePipes.remove(keyPipe);
                                    candidatePipes.add(newCandidatePipe1);
                                    candidatePipes.add(newCandidatePipe2);
                                } else {
                                    // 如果该候选边不在管道库里(端点不可能是删除的keyNode)，增加两个新的虚拟节点（下一轮处理）
                                    // 增加两个新的虚拟节点
                                    VirtualNodeSt newV1 = new VirtualNodeSt();
                                    VirtualNodeSt newV2 = new VirtualNodeSt();
                                    newV1.setDwgId(projectId);
                                    newV1.setUuid(UUID.randomUUID().toString());
                                    newV1.setX(pipe.getStartX());
                                    newV1.setY(pipe.getStartY());
                                    newV2.setDwgId(projectId);
                                    newV2.setUuid(UUID.randomUUID().toString());
                                    newV2.setX(pipe.getEndX());
                                    newV2.setY(pipe.getEndY());
                                    addListV.add(newV1);
                                    addListV.add(newV2);
                                }

                                // 处理候选边库更新
                                KeyPipeSt newPipe1 = pipe.clone();
                                KeyPipeSt newPipe2 = pipe.clone();
                                newPipe1.setEndX(v.getX());
                                newPipe1.setEndY(v.getY());

                                newPipe2.setStartX(v.getX());
                                newPipe2.setStartY(v.getY());

                                iterator2.remove();
                                iterator2.add(newPipe1);
                                iterator2.add(newPipe2);
                            }
                        }
                    }
                }


                // 标记剩下的虚拟节点为最终节点，下次不再处理
                graphProcessor.markFinalNodes(virtualNodeStList);
                // 处理完毕，开始下一轮迭代
                virtualNodeStList.addAll(addListV);
            }

            // Step 3: 剪枝，去除无用的虚拟节点和管道
            while (graphProcessor.checkAloneVirtualNodes(virtualNodeStList, candidatePipes)) {
                ListIterator<VirtualNodeSt> iterator = virtualNodeStList.listIterator();
                while (iterator.hasNext()) {
                    VirtualNodeSt v = iterator.next();
                    List<KeyPipeSt> relatedPipes = graphProcessor.getNodeRelatedPipes(v, candidatePipes);
                    if (relatedPipes.size() == 1) {
                        iterator.remove();
                        candidatePipes.remove(relatedPipes.get(0));
                    }
                }
            }

            // Step 4: 获得Key节点的邻接矩阵结构
            for (InsertSt keyNode : keyNodes) {
                List<InsertSt> upStreamNodes = graphProcessor.getUpStreamNodes(keyNode, candidatePipes, keyNodes);
                List<InsertSt> downStreamNodes = graphProcessor.getDownStreamNodes(keyNode, candidatePipes, keyNodes);
                String upStreamStr = "";
                String downStreamStr = "";
                for (InsertSt node : upStreamNodes) {
                    upStreamStr += node.getInsertHandle0() + "-" + node.getInsertHandle1() + ",";
                }
                for (InsertSt node : downStreamNodes) {
                    downStreamStr += node.getInsertHandle0() + "-" + node.getInsertHandle1() + ",";
                }
                keyNode.setUpstream(upStreamStr);
                keyNode.setDownstream(downStreamStr);
            }

            // Step 5: 入库虚拟节点，关键节点和管道
            for (VirtualNodeSt v : virtualNodeStList) {
                virtualNodeStMapper.insertVirtualNodeSt(v);
            }

            for (InsertSt keyNode : keyNodes) {
                insertStMapper.insertInsertSt(keyNode);
            }

            for (KeyPipeSt pipe : candidatePipes) {
                keyPipeStMapper.insertKeyPipeSt(pipe);
            }

            uploadDwgStMapper.setAnalysed(projectId);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException(e.getMessage());
        }
    }

}
