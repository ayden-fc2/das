package com.dwg.handler.service.impl;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.dao.*;
import com.dwg.handler.entity.*;
import com.dwg.handler.service.DwgService;
import com.dwg.handler.service.feign.FaultFeign;
import com.dwg.handler.utils.GraphProcessor;
import com.dwg.handler.utils.JsonProcessor;
import com.example.common.exception.MyException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;

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
            double errorBase = 0;
            for (InsertSt insertSt : keyNodes) {
                errorBase += insertSt.getBoxWidth() + insertSt.getBoxHeight();
            }
            errorBase = errorBase / keyNodes.size();
            System.out.println("errorBase: " + errorBase);
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
                    int result = graphProcessor.inNodeBox(keyNode, pipe, errorBase);
                    if (result == 0) {
                        // 起点在关键节点内
                        // 入库虚拟节点
                        VirtualNodeSt v = new VirtualNodeSt();
                        v.setDwgId(projectId);
                        v.setVNodeId(-1);
                        v.setUuid(UUID.randomUUID().toString());
                        v.setX(pipe.getEndX());
                        v.setY(pipe.getEndY());
                        String vUUID = graphProcessor.hasVNode(v, virtualNodeStList, errorBase);
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
                        String v1UUID = graphProcessor.hasVNode(v1, virtualNodeStList, errorBase);
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
                            if (graphProcessor.ponitInNodeBox(keyNode, v, errorBase)) {
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
                            int ifVNodeInPipe = graphProcessor.vNodeInLine(v, pipe, errorBase);
                            if (ifVNodeInPipe >= 0) {
                                VirtualNodeSt v1 = graphProcessor.getAnotherVNode(pipe, ifVNodeInPipe, virtualNodeStList, errorBase);
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
                                } else if (graphProcessor.isAnotherNodeDeleted(pipe, ifVNodeInPipe, deleteListV, errorBase)) {
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
                            if (graphProcessor.vNodeOnLineRoad(v, pipe, errorBase)) {
                                KeyPipeSt keyPipe = graphProcessor.isPipeListHasPipe(candidatePipes, pipe, errorBase);
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
                candidatePipes = graphProcessor.uniquePipes(candidatePipes);


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
            Map<String, InsertSt> keyNodeMap = new HashMap<>();
            for (InsertSt node : keyNodes) {
                keyNodeMap.put(node.getInsertHandle0() + "-" + node.getInsertHandle1(), node);
            }
            for (InsertSt keyNode : keyNodes) {
                List<InsertSt> upStreamNodes = graphProcessor.getUpStreamNodes(keyNode, candidatePipes, keyNodes, keyNodeMap);
                List<InsertSt> downStreamNodes = graphProcessor.getDownStreamNodes(keyNode, candidatePipes, keyNodes, keyNodeMap);
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

            // TODO: 入库分析结果
//            uploadDwgStMapper.setAnalysed(projectId);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException(e.getMessage());
        }
    }

    @Autowired
    FaultFeign ff;

    @Override
    public JSONObject genTrace(Long projectId, String faultIds) {
        List<InsertSt> keyNodes = insertStMapper.getInsertStListByDwgId(projectId);
        List<String> faultIdList = Arrays.asList(faultIds.split(","));

        JSONArray nodes = new JSONArray();
        JSONArray edges = new JSONArray();
        Set<String> edgeKeys = new HashSet<>();
        Set<String> allNodeIds = new HashSet<>();

        // ----------------- 第一部分：构建边关系 -----------------
        Map<String, List<String>> adjacencyList = new HashMap<>(); // 邻接表（用于后续中心性计算）
        Map<String, Integer> inDegreeMap = new HashMap<>();       // 入度统计

        for (InsertSt node : keyNodes) {
            String currentNodeId = node.getInsertHandle0() + "-" + node.getInsertHandle1();
            allNodeIds.add(currentNodeId);
            // 处理上游边
            processEdges(node.getUpstream(), currentNodeId, edges, edgeKeys, adjacencyList, inDegreeMap, true);

            // 处理下游边
            processEdges(node.getDownstream(), currentNodeId, edges, edgeKeys, adjacencyList, inDegreeMap, false);
        }

        // ----------------- 第二部分：计算中心性指标 -----------------
        Map<String, Double> betweennessMap = calculateBetweenness(adjacencyList, allNodeIds);  // 中介中心性
        Map<String, Double> closenessMap = calculateCloseness(adjacencyList, allNodeIds);      // 接近中心性

        // 重新计算边和入度
        Map<String, Integer> newInDegreeMap = new HashMap<>();
        JSONArray newEdges = new JSONArray();
        for (Object edge : edges.toArray()) {
            JSONObject newEdge = new JSONObject();
            newEdge.put("source", ((JSONObject) edge).get("target"));
            newEdge.put("target", ((JSONObject) edge).get("source"));
            newEdges.add(newEdge);
        }
        for (InsertSt node : keyNodes) {
            String nodeId = node.getInsertHandle0() + "-" + node.getInsertHandle1();
            newInDegreeMap.put(nodeId, 0);
        }

        // 3. 统一遍历边计算入度（避免嵌套循环）
        for (Object edgeObj : newEdges) {
            JSONObject edge = (JSONObject) edgeObj;
            String target = edge.get("target").toString();
            // 仅统计存在于 keyNodes 中的节点
            if (newInDegreeMap.containsKey(target)) {
                newInDegreeMap.put(target, newInDegreeMap.get(target) + 1);
            }
        }

        // ----------------- 第三部分：构建节点数据 -----------------
        for (InsertSt keyNode : keyNodes) {
            String nodeId = keyNode.getInsertHandle0() + "-" + keyNode.getInsertHandle1();

            JSONObject node = new JSONObject();
            node.put("id", nodeId);
            node.put("x", keyNode.getCenterPtX());
            node.put("y", keyNode.getCenterPtY());
            node.put("is_fault_confidence", 0);
            node.put("is_anomalous_node", faultIdList.contains(nodeId));

            // 添加指标
            node.put("in_degree", newInDegreeMap.getOrDefault(nodeId, 0));
            node.put("betweenness", betweennessMap.getOrDefault(nodeId, 0.0));
            node.put("closeness", closenessMap.getOrDefault(nodeId, 0.0));

            nodes.add(node);
        }

        System.out.println("Edges: " + edges.toJSONString());
        System.out.println("Nodes: " + nodes.toJSONString());
        JSONObject result = new JSONObject();
        result.put("nodes", nodes);
        result.put("edges", newEdges);
        result.put("graph_name", "demo");
        System.out.println(result.toJSONString());

        return ff.getResult(result);
    }

// ----------------- 工具方法 -----------------

    /**
     * 处理边关系并构建邻接表
     * @param edgeStr 边字符串（如"2-956,3-1024"）
     * @param currentNodeId 当前节点ID
     * @param isDownstream 是否处理下游边
     */
    private void processEdges(String edgeStr, String currentNodeId,
                              JSONArray edges, Set<String> edgeKeys,
                              Map<String, List<String>> adjacencyList,
                              Map<String, Integer> inDegreeMap,
                              boolean isDownstream) {
        if (edgeStr == null || edgeStr.isEmpty()) return;

        Arrays.stream(edgeStr.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .forEach(neighborId -> {
                    // 构建边唯一标识
                    String source = isDownstream ? currentNodeId : neighborId;
                    String target = isDownstream ? neighborId : currentNodeId;
                    String edgeKey = source + "->" + target;

                    if (!edgeKeys.contains(edgeKey)) {
                        JSONObject edge = new JSONObject();
                        edge.put("source", source);
                        edge.put("target", target);
                        edges.add(edge);
                        edgeKeys.add(edgeKey);

                        // 更新邻接表
                        adjacencyList.computeIfAbsent(source, k -> new ArrayList<>()).add(target);

                        // 更新入度
                        inDegreeMap.put(target, inDegreeMap.getOrDefault(target, 0) + 1);
                    }
                });
    }

    /**
     * 计算中介中心性（Brandes算法简化实现）
     */
    private Map<String, Double> calculateBetweenness(Map<String, List<String>> adjacencyList, Set<String> allNodes) {
        Map<String, Double> betweenness = new HashMap<>();
        List<String> nodes = new ArrayList<>(allNodes);
        int n = nodes.size();

        for (String s : nodes) {
            Map<String, Integer> distance = new HashMap<>();
            Map<String, Integer> sigma = new HashMap<>();
            Map<String, List<String>> predecessors = new HashMap<>();
            Deque<String> queue = new LinkedList<>();
            Stack<String> stack = new Stack<>();

            // 初始化
            for (String node : nodes) {
                distance.put(node, -1);
                sigma.put(node, 0);
                predecessors.put(node, new ArrayList<>());
            }
            distance.put(s, 0);
            sigma.put(s, 1);
            queue.add(s);

            // BFS遍历计算最短路径
            while (!queue.isEmpty()) {
                String v = queue.poll();
                stack.push(v);
                for (String w : adjacencyList.getOrDefault(v, Collections.emptyList())) {
                    if (distance.get(w) == -1) {
                        distance.put(w, distance.get(v) + 1);
                        queue.add(w);
                    }
                    if (distance.get(w) == distance.get(v) + 1) {
                        sigma.put(w, sigma.get(w) + sigma.get(v));
                        predecessors.get(w).add(v);
                    }
                }
            }

            // 回溯计算delta
            Map<String, Double> delta = new HashMap<>();
            for (String node : nodes) {
                delta.put(node, 0.0);
            }

            while (!stack.isEmpty()) {
                String w = stack.pop();
                if (w.equals(s)) continue;
                for (String v : predecessors.get(w)) {
                    double contrib = (sigma.get(v) * (1.0 + delta.get(w))) / sigma.get(w);
                    delta.put(v, delta.get(v) + contrib);
                }
                betweenness.put(w, betweenness.getOrDefault(w, 0.0) + delta.get(w));
            }
        }

        // 标准化
        if (n > 2) {
            double factor = 1.0 / ((n - 1) * (n - 2));
            betweenness.replaceAll((k, v) -> v * factor);
        }
        return betweenness;
    }


    /**
     * 计算接近中心性（基于BFS）
     */
    private Map<String, Double> calculateCloseness(Map<String, List<String>> adjacencyList, Set<String> allNodes) {
        Map<String, Double> closeness = new HashMap<>();
        int totalNodes = allNodes.size(); // 总节点数

        // 遍历所有节点
        for (String node : allNodes) {
            Map<String, Integer> distance = new HashMap<>();
            Deque<String> queue = new LinkedList<>();
            distance.put(node, 0);
            queue.offer(node);

            // BFS 遍历图，计算每个节点到当前节点的最短路径
            while (!queue.isEmpty()) {
                String current = queue.poll();
                for (String neighbor : adjacencyList.getOrDefault(current, Collections.emptyList())) {
                    if (!distance.containsKey(neighbor)) {
                        distance.put(neighbor, distance.get(current) + 1);
                        queue.offer(neighbor);
                    }
                }
            }

            // 计算可达节点数量和总距离
            int totalDistance = 0;
            int reachableNodes = 0;
            for (Map.Entry<String, Integer> entry : distance.entrySet()) {
                if (!entry.getKey().equals(node)) { // 排除自己
                    totalDistance += entry.getValue();
                    reachableNodes++;
                }
            }

            // 如果节点不可达，接近中心性设为 0
            double closenessCentrality = 0.0;
            if (reachableNodes > 0 && totalDistance > 0) {
                // 正常情况下的接近中心性计算
                closenessCentrality = (double) (reachableNodes) / totalDistance;

                // 如果图是多连通分量的, 使用改进公式
                if (totalNodes > 1) {
                    // 计算图的接近中心性标准化 (Wasserman 和 Faust 改进公式)
                    closenessCentrality *= (double) (reachableNodes) / (totalNodes - 1);
                }
            }

            // 存储接近中心性
            closeness.put(node, closenessCentrality);
        }

        return closeness;
    }




}
