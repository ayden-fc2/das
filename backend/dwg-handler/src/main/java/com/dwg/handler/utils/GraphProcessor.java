package com.dwg.handler.utils;

import com.dwg.handler.entity.InsertSt;
import com.dwg.handler.entity.KeyPipeSt;
import com.dwg.handler.entity.VirtualNodeSt;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Component
public class GraphProcessor {
    // 设置误差
    private static final double ERROR = 0.01;

    /**
     * 判断节点是否在节点框内
     *
     * @param keyNode
     * @param v
     * @return
     */
    public boolean ponitInNodeBox(InsertSt insertSt, VirtualNodeSt v) {
        double boxMaxX = insertSt.getBoxWidth() / 2 + insertSt.getCenterPtX() + ERROR;
        double boxMaxY = insertSt.getBoxHeight() / 2 + insertSt.getCenterPtY() + ERROR;
        double boxMinX = insertSt.getCenterPtX() - insertSt.getBoxWidth() / 2 - ERROR;
        double boxMinY = insertSt.getCenterPtY() - insertSt.getBoxHeight() / 2 - ERROR;
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
    public int inNodeBox(InsertSt insertSt, KeyPipeSt keyPipeSt) {
        int result = -1;
        double boxMaxX = insertSt.getBoxWidth() / 2 + insertSt.getCenterPtX() + ERROR;
        double boxMaxY = insertSt.getBoxHeight() / 2 + insertSt.getCenterPtY() + ERROR;
        double boxMinX = insertSt.getCenterPtX() - insertSt.getBoxWidth() / 2 - ERROR;
        double boxMinY = insertSt.getCenterPtY() - insertSt.getBoxHeight() / 2 - ERROR;
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
    public String hasVNode(VirtualNodeSt v1, List<VirtualNodeSt> virtualNodeStList) {
        for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
            if (Math.abs(virtualNodeSt.getX() - v1.getX()) <= ERROR && Math.abs(virtualNodeSt.getY() - v1.getY()) <= ERROR) {
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
     * 对管道进行去重
     *
     * @param candidatePipes
     * @return
     */
    public List<KeyPipeSt> uniquePipes(List<KeyPipeSt> candidatePipes) {
        Set<String> seenKeys = new HashSet<>();
        List<KeyPipeSt> uniqueList = new ArrayList<>();

        for (KeyPipeSt pipe : candidatePipes) {
            // 构造组合键（注意处理字符串可能为 null 的情况）
            String key = pipe.getStartKeyHandle0() + "_" +
                    pipe.getStartKeyHandle1() + "_" +
                    pipe.getEndKeyHandle0() + "_" +
                    pipe.getEndKeyHandle1() + "_" +
                    (pipe.getVStartUUID() == null ? "" : pipe.getVStartUUID()) + "_" +
                    (pipe.getVEndUUID() == null ? "" : pipe.getVEndUUID());

            if (!seenKeys.contains(key)) {
                seenKeys.add(key);
                uniqueList.add(pipe);
            }
        }

        return uniqueList;
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
    public int vNodeInLine(VirtualNodeSt v, KeyPipeSt pipe) {
        double x = v.getX();
        double y = v.getY();
        double pipeStartX = pipe.getStartX();
        double pipeStartY = pipe.getStartY();
        double pipeEndX = pipe.getEndX();
        double pipeEndY = pipe.getEndY();
        if (Math.abs(x - pipeStartX) <= ERROR && Math.abs(y - pipeStartY) <= ERROR) {
            return 0;
        } else if (Math.abs(x - pipeEndX) <= ERROR && Math.abs(y - pipeEndY) <= ERROR) {
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
    public VirtualNodeSt getAnotherVNode(KeyPipeSt pipe, int ifVNodeInPipe, List<VirtualNodeSt> virtualNodeStList) {
        if (ifVNodeInPipe == 0) {
            // 已有节点在起点
            for (VirtualNodeSt virtualNodeSt : virtualNodeStList) {
                double x = virtualNodeSt.getX();
                double y = virtualNodeSt.getY();
                double pipeEndX = pipe.getEndX();
                double pipeEndY = pipe.getEndY();
                if (Math.abs(x - pipeEndX) <= ERROR && Math.abs(y - pipeEndY) <= ERROR) {
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
                if (Math.abs(x - pipeStartX) <= ERROR && Math.abs(y - pipeStartY) <= ERROR) {
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
    public boolean isAnotherNodeDeleted(KeyPipeSt pipe, int ifVNodeInPipe, List<VirtualNodeSt> deleteListV) {
        if (ifVNodeInPipe == 0) {
            // 已有节点在起点
            double pipeEndX = pipe.getEndX();
            double pipeEndY = pipe.getEndY();
            for (VirtualNodeSt virtualNodeSt : deleteListV) {
                double x = virtualNodeSt.getX();
                double y = virtualNodeSt.getY();
                if (Math.abs(x - pipeEndX) <= ERROR && Math.abs(y - pipeEndY) <= ERROR) {
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
                if (Math.abs(x - pipeStartX) <= ERROR && Math.abs(y - pipeStartY) <= ERROR) {
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
    public boolean vNodeOnLineRoad(VirtualNodeSt v, KeyPipeSt pipe) {
        double px = v.getX();
        double py = v.getY();
        double ax = pipe.getStartX();
        double ay = pipe.getStartY();
        double bx = pipe.getEndX();
        double by = pipe.getEndY();

        // 检查是否在起点或终点的误差范围内
        if (isWithinError(px, py, ax, ay) || isWithinError(px, py, bx, by)) {
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

        return distanceSq <= ERROR * ERROR;
    }

    // 判断点(p)是否在点(target)的误差范围内
    private boolean isWithinError(double px, double py, double tx, double ty) {
        double dx = px - tx;
        double dy = py - ty;
        return dx * dx + dy * dy <= ERROR * ERROR;
    }


    /**
     * 判断管道库里是否有该管道
     * @param candidatePipes
     * @param pipe
     * @return
     */
    public KeyPipeSt isPipeListHasPipe(List<KeyPipeSt> candidatePipes, KeyPipeSt pipe) {
        double pipeStartX = pipe.getStartX();
        double pipeStartY = pipe.getStartY();
        double pipeEndX = pipe.getEndX();
        double pipeEndY = pipe.getEndY();
        for (KeyPipeSt candidatePipe : candidatePipes) {
            if (Math.abs(candidatePipe.getStartX() - pipeStartX) <= ERROR
                    && Math.abs(candidatePipe.getStartY() - pipeStartY) <= ERROR
                    && Math.abs(candidatePipe.getEndX() - pipeEndX) <= ERROR
                    && Math.abs(candidatePipe.getEndY() - pipeEndY) <= ERROR) {
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
    public List<InsertSt> getUpStreamNodes(InsertSt keyNode, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes) {
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
                    List<InsertSt> upStreamNodes = getUpStreamNodesFromVNode(candidatePipe.getVStartUUID(), candidatePipes, keyNodes);
                    result.addAll(upStreamNodes);
                }
            }
        }
        return result;
    }

    List<InsertSt> getUpStreamNodesFromVNode(String uuid, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes) {
        List<InsertSt> result = new ArrayList<>();
        for (KeyPipeSt candidatePipe : candidatePipes) {
            String pUUID = candidatePipe.getVEndUUID() == null ? "" : candidatePipe.getVEndUUID();
            if (pUUID.equals(uuid)) {
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
                    List<InsertSt> upStreamNodes = getUpStreamNodesFromVNode(candidatePipe.getVStartUUID(), candidatePipes, keyNodes);
                    result.addAll(upStreamNodes);
                }
            }
        }
        return result;
    }



    /**
     * 获取某节点的下游节点（直接）
     * @param keyNode
     * @param candidatePipes
     * @param keyNodes
     * @return
     */
    public List<InsertSt> getDownStreamNodes(InsertSt keyNode, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes) {
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
                    List<InsertSt> downStreamNodes = getDownStreamNodesFromVNode(candidatePipe.getVEndUUID(), candidatePipes, keyNodes);
                    result.addAll(downStreamNodes);
                }
            }
        }
        return result;
    }

    List<InsertSt> getDownStreamNodesFromVNode(String uuid, List<KeyPipeSt> candidatePipes, List<InsertSt> keyNodes) {
        List<InsertSt> result = new ArrayList<>();
        for (KeyPipeSt candidatePipe : candidatePipes) {
            String pUUID = candidatePipe.getVStartUUID() == null ? "" : candidatePipe.getVStartUUID();
            if (pUUID.equals(uuid)) {
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
                    List<InsertSt> downStreamNodes = getDownStreamNodesFromVNode(candidatePipe.getVEndUUID(), candidatePipes, keyNodes);
                    result.addAll(downStreamNodes);
                }
            }
        }
        return result;
    }
}
