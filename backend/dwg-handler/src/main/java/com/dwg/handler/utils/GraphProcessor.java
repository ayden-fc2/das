package com.dwg.handler.utils;

import com.dwg.handler.entity.InsertSt;
import com.dwg.handler.entity.KeyPipeSt;
import com.dwg.handler.entity.VirtualNodeSt;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

@Component
public class GraphProcessor {
    // 设置误差
    private static final double ERROR = 0.00005;

    /**
     * 判断节点是否在节点框内
     *
     * @param
     * @param v
     * @return
     */
    public boolean ponitInNodeBox(InsertSt insertSt, VirtualNodeSt v, double errorBase) {
        double boxMaxX = insertSt.getBoxWidth() / 2 + insertSt.getCenterPtX() + ERROR * errorBase;
        double boxMaxY = insertSt.getBoxHeight() / 2 + insertSt.getCenterPtY() + ERROR * errorBase;
        double boxMinX = insertSt.getCenterPtX() - insertSt.getBoxWidth() / 2 - ERROR * errorBase;
        double boxMinY = insertSt.getCenterPtY() - insertSt.getBoxHeight() / 2 - ERROR * errorBase;
        if (
                v.getX() <= boxMaxX
                        && v.getX() >= boxMinX
                        && v.getY() <= boxMaxY
                        && v.getY() >= boxMinY
        ) {
            return true;
        }
        return false;
    }

    /**
     * 传入管道，判断节点是否在节点框内
     *
     * @param insertSt
     * @param keyPipeSt
     * @return -1: 不在节点内， 0: 起点在节点内， 1: 终点在节点内
     */
    public int inNodeBox(InsertSt insertSt, KeyPipeSt keyPipeSt, double errorBase) {
        int result = -1;
        double boxMaxX = insertSt.getBoxWidth() / 2 + insertSt.getCenterPtX() + ERROR * errorBase;
        double boxMaxY = insertSt.getBoxHeight() / 2 + insertSt.getCenterPtY() + ERROR * errorBase;
        double boxMinX = insertSt.getCenterPtX() - insertSt.getBoxWidth() / 2 - ERROR * errorBase;
        double boxMinY = insertSt.getCenterPtY() - insertSt.getBoxHeight() / 2 - ERROR * errorBase;
        if (
                keyPipeSt.getStartX() <= boxMaxX
                        && keyPipeSt.getStartX() >= boxMinX
                        && keyPipeSt.getStartY() <= boxMaxY
                        && keyPipeSt.getStartY() >= boxMinY
        ) {
            result = 0;
        } else if (
                keyPipeSt.getEndX() <= boxMaxX
                        && keyPipeSt.getEndX() >= boxMinX
                        && keyPipeSt.getEndY() <= boxMaxY
                        && keyPipeSt.getEndY() >= boxMinY
        ) {
            result = 1;
        }
        return result;
    }


    /**
     * 判断是否继续迭代处理
     *
     * @param virtualNodeStList
     * @return
     */
    public boolean checkVirtualNodes(List<VirtualNodeSt> virtualNodeStList) {
        for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
            if (virtualNodeSt.getFinished() != 1) {
                return true;
            }
        }
        return false;
    }

    /**
     * 判断虚拟节点列表中是否已经有该节点，避免重复处理
     *
     * @param v1
     * @param virtualNodeStList
     * @return
     */
    public String hasVNode(VirtualNodeSt v1, List<VirtualNodeSt> virtualNodeStList, double errorBase) {
        for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
            if (Math.abs(virtualNodeSt.getX() - v1.getX()) <= (ERROR * errorBase) && Math.abs(virtualNodeSt.getY() - v1.getY()) <= (ERROR * errorBase)) {
                return virtualNodeSt.getUuid();
            }
        }
        return null;
    }

    /**
     * 更新虚拟节点在关键节点内，进行替换
     *
     * @param v
     * @param keyNode
     * @param candidatePipes
     * @param virtualNodeStList
     */
    public void updateV2Key(VirtualNodeSt v, InsertSt keyNode, List<KeyPipeSt> candidatePipes, List<VirtualNodeSt> virtualNodeStList) {
        // 对管道替换
        for (KeyPipeSt keyPipeSt : candidatePipes) {
            String startUUID = keyPipeSt.getVStartUUID() == null ? "" : keyPipeSt.getVStartUUID();
            String endUUID = keyPipeSt.getVEndUUID() == null ? "" : keyPipeSt.getVEndUUID();
            if (startUUID.equals(v.getUuid())) {
                keyPipeSt.setVStartUUID(null);
                keyPipeSt.setStartKeyHandle0(keyNode.getInsertHandle0());
                keyPipeSt.setStartKeyHandle1(keyNode.getInsertHandle1());
            } else if (endUUID.equals(v.getUuid())) {
                keyPipeSt.setVEndUUID(null);
                keyPipeSt.setEndKeyHandle0(keyNode.getInsertHandle0());
                keyPipeSt.setEndKeyHandle1(keyNode.getInsertHandle1());
            }
        }
    }

    /**
     * BFS 对管道进行去重
     *
     * @param candidatePipes
     * @return
     */
    public List<KeyPipeSt> uniquePipes(List<KeyPipeSt> candidatePipes) {
        // 第一步：去重环和完全相同的边
        Set<String> seenKeys = new HashSet<>();
        List<KeyPipeSt> filteredList = new ArrayList<>();
        for (KeyPipeSt pipe : candidatePipes) {
            String startKey = pipe.getStartKeyHandle0() + "_" + pipe.getStartKeyHandle1() + "_" + pipe.getVStartUUID();
            String endKey = pipe.getEndKeyHandle0() + "_" + pipe.getEndKeyHandle1() + "_" + pipe.getVEndUUID();
            if (startKey.equals(endKey)) continue;

            String uniqueKey = pipe.getStartKeyHandle0() + "_"
                    + pipe.getStartKeyHandle1() + "_"
                    + pipe.getEndKeyHandle0() + "_"
                    + pipe.getEndKeyHandle1() + "_"
                    + (pipe.getVStartUUID() == null ? "" : pipe.getVStartUUID()) + "_"
                    + (pipe.getVEndUUID() == null ? "" : pipe.getVEndUUID());
            if (!seenKeys.contains(uniqueKey)) {
                seenKeys.add(uniqueKey);
                filteredList.add(pipe);
            }
        }

        // 第二步：构建邻接表
        Map<String, List<String>> adjacencyList = new HashMap<>();
        for (KeyPipeSt pipe : filteredList) {
            String startKey = pipe.getStartKeyHandle0() + "_" + pipe.getStartKeyHandle1() + "_" + pipe.getVStartUUID();
            String endKey = pipe.getEndKeyHandle0() + "_" + pipe.getEndKeyHandle1() + "_" + pipe.getVEndUUID();
            adjacencyList.computeIfAbsent(startKey, k -> new ArrayList<>()).add(endKey);
        }

        // 第三步：遍历所有大边，并检查是否存在替代路径
        List<KeyPipeSt> pipesToRemove = new ArrayList<>();
        for (KeyPipeSt pipe : filteredList) {
            String startKey = pipe.getStartKeyHandle0() + "_" + pipe.getStartKeyHandle1() + "_" + pipe.getVStartUUID();
            String endKey = pipe.getEndKeyHandle0() + "_" + pipe.getEndKeyHandle1() + "_" + pipe.getVEndUUID();

            // 判断是否为大边（起点/终点为真实节点）
            boolean isStartReal = (pipe.getVStartUUID() == null || pipe.getVStartUUID().isEmpty());
            boolean isEndReal = (pipe.getVEndUUID() == null || pipe.getVEndUUID().isEmpty());
            if (isStartReal || isEndReal) {
                // 检查是否存在通过虚拟节点的替代路径
                boolean hasAlternative = checkAlternativePath(startKey, endKey, adjacencyList);
                if (hasAlternative) {
                    pipesToRemove.add(pipe);
                }
            }
        }

        // 第四步：移除需要去除的大边
        filteredList.removeAll(pipesToRemove);

        return filteredList;
    }

    private boolean checkAlternativePath(String start, String end, Map<String, List<String>> adjacencyList) {
        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        queue.offer(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            List<String> neighbors = adjacencyList.get(current);
            if (neighbors == null) continue;

            for (String neighbor : neighbors) {
                // 排除当前大边本身的直接连接
                if (current.equals(start) && neighbor.equals(end)) {
                    continue;
                }

                if (neighbor.equals(end)) {
                    return true; // 找到替代路径
                }

                // 检查是否为虚拟节点（UUID非空）
                String[] parts = neighbor.split("_");
                if (parts.length < 3) continue; // 格式错误处理
                String uuid = parts[2];
                if (!uuid.isEmpty() && !visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        return false;
    }


    /**
     * 标记最终节点
     *
     * @param virtualNodeStList
     */
    public void markFinalNodes(List<VirtualNodeSt> virtualNodeStList) {
        for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
            virtualNodeSt.setFinished(1);
        }
    }

    /**
     * 判断虚拟节点是否在管道中
     *
     * @param v
     * @param pipe
     * @return -1 不在 0 起点 1 终点
     */
    public int vNodeInLine(VirtualNodeSt v, KeyPipeSt pipe, double errorBase) {
        double x = v.getX();
        double y = v.getY();
        double pipeStartX = pipe.getStartX();
        double pipeStartY = pipe.getStartY();
        double pipeEndX = pipe.getEndX();
        double pipeEndY = pipe.getEndY();
        if (Math.abs(x - pipeStartX) <= (ERROR * errorBase) && Math.abs(y - pipeStartY) <= (ERROR * errorBase)) {
            return 0;
        } else if (Math.abs(x - pipeEndX) <= (ERROR * errorBase) && Math.abs(y - pipeEndY) <= (ERROR * errorBase)) {
            return 1;
        }
        return -1;
    }

    /**
     * 获取另一个虚拟节点
     * @param v
     * @param pipe
     * @param ifVNodeInPipe
     * @param virtualNodeStList
     * @return
     */
    public VirtualNodeSt getAnotherVNode(KeyPipeSt pipe, int ifVNodeInPipe, List<VirtualNodeSt> virtualNodeStList, double errorBase) {
        if (ifVNodeInPipe == 0) {
            // 已有节点在起点
            for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
                double x = virtualNodeSt.getX();
                double y = virtualNodeSt.getY();
                double pipeEndX = pipe.getEndX();
                double pipeEndY = pipe.getEndY();
                if (Math.abs(x - pipeEndX) <= (ERROR * errorBase) && Math.abs(y - pipeEndY) <= (ERROR * errorBase)) {
                    return virtualNodeSt;
                }
            }
        } else if (ifVNodeInPipe == 1) {
            // 已有节点在终点
            for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
                double x = virtualNodeSt.getX();
                double y = virtualNodeSt.getY();
                double pipeStartX = pipe.getStartX();
                double pipeStartY = pipe.getStartY();
                if (Math.abs(x - pipeStartX) <= (ERROR * errorBase) && Math.abs(y - pipeStartY) <= (ERROR * errorBase)) {
                    return virtualNodeSt;
                }
            }
        }
        return null;
    }

    /**
     * 判断另一个节点是否被删除
     * @param v
     * @param pipe
     * @param ifVNodeInPipe
     * @param deleteListV
     * @return
     */
    public boolean isAnotherNodeDeleted(KeyPipeSt pipe, int ifVNodeInPipe, List<VirtualNodeSt> deleteListV, double errorBase) {
        if (ifVNodeInPipe == 0) {
            // 已有节点在起点
            double pipeEndX = pipe.getEndX();
            double pipeEndY = pipe.getEndY();
            for (VirtualNodeSt virtualNodeSt : deleteListV) {
                double x = virtualNodeSt.getX();
                double y = virtualNodeSt.getY();
                if (Math.abs(x - pipeEndX) <= (ERROR * errorBase) && Math.abs(y - pipeEndY) <= (ERROR * errorBase)) {
                    return true;
                }
            }
        } else {
            double pipeStartX = pipe.getStartX();
            double pipeStartY = pipe.getStartY();
            // 已有节点在终点
            for (VirtualNodeSt virtualNodeSt : deleteListV) {
                double x = virtualNodeSt.getX();
                double y = virtualNodeSt.getY();
                if (Math.abs(x - pipeStartX) <= (ERROR * errorBase) && Math.abs(y - pipeStartY) <= (ERROR * errorBase)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 判断虚拟节点是否在管道路径中
     * @param v
     * @param pipe
     * @return
     */
    public boolean vNodeOnLineRoad(VirtualNodeSt v, KeyPipeSt pipe, double errorBase) {
        double px = v.getX();
        double py = v.getY();
        double ax = pipe.getStartX();
        double ay = pipe.getStartY();
        double bx = pipe.getEndX();
        double by = pipe.getEndY();

        // 检查是否在起点或终点的误差范围内
        if (isWithinError(px, py, ax, ay, errorBase) || isWithinError(px, py, bx, by, errorBase)) {
            return false;
        }

        // 计算线段向量和长度平方
        double abx = bx - ax;
        double aby = by - ay;
        double abLengthSq = abx * abx + aby * aby;

        // 处理线段退化为点的情况
        if (abLengthSq < 1e-20) { // 极小值判断，避免浮点误差
            return false;
        }

        // 计算投影参数t
        double apx = px - ax;
        double apy = py - ay;
        double t = (apx * abx + apy * aby) / abLengthSq;

        // 投影点不在线段上
        if (t < 0.0 || t > 1.0) {
            return false;
        }

        // 计算投影点坐标
        double projX = ax + t * abx;
        double projY = ay + t * aby;

        // 计算点到投影点的距离平方
        double dx = px - projX;
        double dy = py - projY;
        double distanceSq = dx * dx + dy * dy;

        return distanceSq <= (ERROR * errorBase) * (ERROR * errorBase);
    }

    // 判断点(p)是否在点(target)的误差范围内
    private boolean isWithinError(double px, double py, double tx, double ty, double errorBase) {
        double dx = px - tx;
        double dy = py - ty;
        return dx * dx + dy * dy <= (ERROR * errorBase) * (ERROR * errorBase);
    }


    /**
     * 判断管道库里是否有该管道
     * @param candidatePipes
     * @param pipe
     * @return
     */
    public KeyPipeSt isPipeListHasPipe(List<KeyPipeSt> candidatePipes, KeyPipeSt pipe, double errorBase) {
        double pipeStartX = pipe.getStartX();
        double pipeStartY = pipe.getStartY();
        double pipeEndX = pipe.getEndX();
        double pipeEndY = pipe.getEndY();
        for (KeyPipeSt candidatePipe : candidatePipes) {
            if (Math.abs(candidatePipe.getStartX() - pipeStartX) <= (ERROR * errorBase)
                    && Math.abs(candidatePipe.getStartY() - pipeStartY) <= (ERROR * errorBase)
                    && Math.abs(candidatePipe.getEndX() - pipeEndX) <= (ERROR * errorBase)
                    && Math.abs(candidatePipe.getEndY() - pipeEndY) <= (ERROR * errorBase)) {
                return candidatePipe;
            }
        }
        return null;
    }

    /**
     * 检查有无孤立的节点
     * @param virtualNodeStList
     * @param candidatePipes
     * @return
     */
    public boolean checkAloneVirtualNodes(List<VirtualNodeSt> virtualNodeStList, List<KeyPipeSt> candidatePipes) {
        for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
            List<KeyPipeSt> relatedPipes = getNodeRelatedPipes(virtualNodeSt, candidatePipes);
            // 符合相关管道只有一个时，说明存在孤立节点
            if (relatedPipes.size() == 1) {
                return true;
            }
        }
        return false;
    }

    /**
     * 判断某节点在所有管道中是否孤立
     * @param v
     * @param candidatePipes
     * @return
     */
    public List<KeyPipeSt> getNodeRelatedPipes(VirtualNodeSt v, List<KeyPipeSt> candidatePipes) {
        List<KeyPipeSt> relatedPipes = new ArrayList<>();
        for (KeyPipeSt candidatePipe : candidatePipes) {
            String vId = v.getUuid();
            String pStartId = candidatePipe.getVStartUUID() == null ? "" : candidatePipe.getVStartUUID();
            String pEndId = candidatePipe.getVEndUUID() == null ? "" : candidatePipe.getVEndUUID();
            if (vId.equals(pStartId) || vId.equals(pEndId)) {
                relatedPipes.add(candidatePipe);
            }
        }
        return relatedPipes;
    }

    /**
     * 获取某节点的上游节点（直接）
     * @param keyNode
     * @param candidatePipes
     * @param keyNodes
     * @return
     */
    public List<InsertSt> getUpStreamNodes(InsertSt keyNode, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes, Map<String, InsertSt> keyNodeMap) {
        List<InsertSt> result = new ArrayList<>();
        for (KeyPipeSt candidatePipe : candidatePipes) {
            if (candidatePipe.getEndKeyHandle0()==keyNode.getInsertHandle0() && candidatePipe.getEndKeyHandle1()==keyNode.getInsertHandle1()) {
                // 定位相关管道
                if (candidatePipe.getVStartUUID() == null) {
                    // 找到了一条key节点
                    for (InsertSt keyNode1 : keyNodes) {
                        if (keyNode1.getInsertHandle0() == candidatePipe.getStartKeyHandle0() && keyNode1.getInsertHandle1() == candidatePipe.getStartKeyHandle1()) {
                            result.add(keyNode1);
                            break;
                        }
                    }
                } else {
                    // 找到了一条虚拟节点
                    List<InsertSt> upStreamNodes = getUpStreamNodesFromVNode(candidatePipe.getVStartUUID(), candidatePipes, keyNodeMap);
                    result.addAll(upStreamNodes);
                }
            }
        }
        return result;
    }

    List<InsertSt> getUpStreamNodesFromVNode(String uuid, List<KeyPipeSt> candidatePipes, Map<String, InsertSt> keyNodeMap) {
        List<InsertSt> result = new ArrayList<>();
        Deque<String> stack = new ArrayDeque<>();
        Set<String> visited = new HashSet<>();
        stack.push(uuid);

        while (!stack.isEmpty()) {
            String currentUuid = stack.pop();
            if (!visited.add(currentUuid)) continue; // 已访问则跳过

            for (KeyPipeSt pipe : candidatePipes) {
                String pipeVEnd = pipe.getVEndUUID() != null ? pipe.getVEndUUID() : "";
                if (!pipeVEnd.equals(currentUuid)) continue;

                if (pipe.getVStartUUID() == null) {
                    // 直接连接到KeyNode
                    String key = pipe.getStartKeyHandle0() + "-" + pipe.getStartKeyHandle1();
                    InsertSt node = keyNodeMap.get(key);
                    if (node != null) result.add(node);
                } else {
                    // 虚拟节点，继续遍历
                    if (!visited.contains(pipe.getVStartUUID())) {
                        stack.push(pipe.getVStartUUID());
                    }
                }
            }
        }
        return result;
    }

//    List<InsertSt> getUpStreamNodesFromVNode(String uuid, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes) {
//        List<InsertSt> result = new ArrayList<>();
//        for (KeyPipeSt candidatePipe : candidatePipes) {
//            String pUUID = candidatePipe.getVEndUUID() == null ? "" : candidatePipe.getVEndUUID();
//            if (pUUID.equals(uuid)) {
//                // 定位相关管道
//                if (candidatePipe.getVStartUUID() == null) {
//                    // 找到了一条key节点
//                    for (InsertSt keyNode1 : keyNodes) {
//                        if (keyNode1.getInsertHandle0() == candidatePipe.getStartKeyHandle0() && keyNode1.getInsertHandle1() == candidatePipe.getStartKeyHandle1()) {
//                            result.add(keyNode1);
//                            break;
//                        }
//                    }
//                } else {
//                    // 找到了一条虚拟节点
//                    List<InsertSt> upStreamNodes = getUpStreamNodesFromVNode(candidatePipe.getVStartUUID(), candidatePipes, keyNodes);
//                    result.addAll(upStreamNodes);
//                }
//            }
//        }
//        return result;
//    }



    /**
     * 获取某节点的下游节点（直接）
     * @param keyNode
     * @param candidatePipes
     * @param keyNodes
     * @return
     */
    public List<InsertSt> getDownStreamNodes(InsertSt keyNode, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes, Map<String, InsertSt> keyNodeMap) {
        List<InsertSt> result = new ArrayList<>();
        for (KeyPipeSt candidatePipe : candidatePipes) {
            if (candidatePipe.getStartKeyHandle0()==keyNode.getInsertHandle0() && candidatePipe.getStartKeyHandle1()==keyNode.getInsertHandle1()) {
                // 定位相关管道
                if (candidatePipe.getVEndUUID() == null) {
                    // 找到了一条key节点
                    for (InsertSt keyNode1 : keyNodes) {
                        if (keyNode1.getInsertHandle0() == candidatePipe.getEndKeyHandle0() && keyNode1.getInsertHandle1() == candidatePipe.getEndKeyHandle1()) {
                            result.add(keyNode1);
                            break;
                        }
                    }
                } else {
                    // 找到了一条虚拟节点
                    List<InsertSt> downStreamNodes = getDownStreamNodesFromVNode(candidatePipe.getVEndUUID(), candidatePipes, keyNodeMap);
                    result.addAll(downStreamNodes);
                }
            }
        }
        return result;
    }

    List<InsertSt> getDownStreamNodesFromVNode(String uuid, List<KeyPipeSt> candidatePipes, Map<String, InsertSt> keyNodeMap) {
        List<InsertSt> result = new ArrayList<>();
        Deque<String> stack = new ArrayDeque<>();
        Set<String> visited = new HashSet<>();
        stack.push(uuid);

        while (!stack.isEmpty()) {
            String currentUuid = stack.pop();
            if (!visited.add(currentUuid)) continue;

            for (KeyPipeSt pipe : candidatePipes) {
                String pipeVStart = pipe.getVStartUUID() != null ? pipe.getVStartUUID() : "";
                if (!pipeVStart.equals(currentUuid)) continue;

                if (pipe.getVEndUUID() == null) {
                    // 直接连接到KeyNode
                    String key = pipe.getEndKeyHandle0() + "-" + pipe.getEndKeyHandle1();
                    InsertSt node = keyNodeMap.get(key);
                    if (node != null) result.add(node);
                } else {
                    // 虚拟节点，继续遍历
                    if (!visited.contains(pipe.getVEndUUID())) {
                        stack.push(pipe.getVEndUUID());
                    }
                }
            }
        }
        return result;
    }

//    List<InsertSt> getDownStreamNodesFromVNode(String uuid, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes) {
//        List<InsertSt> result = new ArrayList<>();
//        for (KeyPipeSt candidatePipe : candidatePipes) {
//            String pUUID = candidatePipe.getVStartUUID() == null ? "" : candidatePipe.getVStartUUID();
//            if (pUUID.equals(uuid)) {
//                // 定位相关管道
//                if (candidatePipe.getVEndUUID() == null) {
//                    // 找到了一条key节点
//                    for (InsertSt keyNode1 : keyNodes) {
//                        if (keyNode1.getInsertHandle0() == candidatePipe.getEndKeyHandle0() && keyNode1.getInsertHandle1() == candidatePipe.getEndKeyHandle1()) {
//                            result.add(keyNode1);
//                            break;
//                        }
//                    }
//                } else {
//                    // 找到了一条虚拟节点
//                    List<InsertSt> downStreamNodes = getDownStreamNodesFromVNode(candidatePipe.getVEndUUID(), candidatePipes, keyNodes);
//                    result.addAll(downStreamNodes);
//                }
//            }
//        }
//        return result;
//    }
}
