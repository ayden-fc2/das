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
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

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
            // 分别入库 blockData, insertsData
            blockStMapper.deleteByDwgId(projectId);
            insertStMapper.deleteByDwgId(projectId);
            keyPipeStMapper.deleteByDwgId(projectId);
            virtualNodeStMapper.deleteByDwgId(projectId);
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
                insertStMapper.insertInsertSt(insertSt);
                keyNodes.add(insertSt);
            }
            // TODO: 图算法处理pipesData，得到graph结构，以及insert间的上下游关系
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
            List<VirtualNodeSt> nextRound = new ArrayList<>();
            while (graphProcessor.checkVirtualNodes(virtualNodeStList)) {
                // 先处理所有的虚拟节点，如果在关键节点内，则更新相关管道，删除虚拟节点，最后对管道去重
                List<VirtualNodeSt> deleteList = new ArrayList<>();
                for (VirtualNodeSt v : virtualNodeStList) {
                    for (InsertSt keyNode : keyNodes) {
                        if (graphProcessor.ponitInNodeBox(keyNode, v)) {
                            graphProcessor.updateV2Key(v, keyNode, candidatePipes, virtualNodeStList);
                            deleteList.add(v);
                        }
                    }
                }
                virtualNodeStList.removeAll(deleteList);
                candidatePipes = graphProcessor.uniquePipes(candidatePipes);
                // 对剩下的虚拟节点去寻找候选边，如果在候选边的端点上，则增加管道，分情况讨论
                // 如果候选边的另一个端点是已有的虚拟节点，则处理管道，去重
                // 如果候选边的另一个端点是未知节点，则创建新的虚拟节点以及新的管道

                // 对剩下的虚拟节点，查看它是否在候选边的路径上，如果在，则分情况讨论
                // 如果该候选边已经在管道库里，继承该管道的属性，拆成两个管道更新管道库，连同候选边库一同更新，对节点进行进一步处理
                // 如果该候选边不在管道库里，将该候选边拆成两个管道入库，连同候选边库一同更新，增加两个新的虚拟节点

                // 标记剩下的虚拟节点为最终节点，下次不再处理
                graphProcessor.markFinalNodes(virtualNodeStList);
                // 处理完毕，开始下一轮迭代
                virtualNodeStList.addAll(nextRound);
            }

            // Step 3: 剪枝，去除无用的虚拟节点和管道


            // Step 4: 入库虚拟节点和管道
            for (VirtualNodeSt v : virtualNodeStList) {
                virtualNodeStMapper.insertVirtualNodeSt(v);
            }

            for (KeyPipeSt pipe : candidatePipes) {
                keyPipeStMapper.insertKeyPipeSt(pipe);
            }

            // uploadDwgStMapper.setAnalysed(projectId);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException(e.getMessage());
        }
    }

}
