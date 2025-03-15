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
     * @param virtualNodeStList
     * @return
     */
    public boolean checkVirtualNodes(List<VirtualNodeSt> virtualNodeStList) {
        for (VirtualNodeSt virtualNodeSt: virtualNodeStList) {
            if (virtualNodeSt.getFinished() != 1) {
                return true;
            }
        }
        return false;
    }

    /**
     * 判断虚拟节点列表中是否已经有该节点，避免重复处理
     * @param v1
     * @param virtualNodeStList
     * @return
     */
    public String hasVNode(VirtualNodeSt v1, List<VirtualNodeSt> virtualNodeStList) {
        for (VirtualNodeSt virtualNodeSt: virtualNodeStList) {
            if (Math.abs(virtualNodeSt.getX() - v1.getX()) <= ERROR && Math.abs(virtualNodeSt.getY() - v1.getY()) <= ERROR) {
                return virtualNodeSt.getUuid();
            }
        }
        return null;
    }

    /**
     * 更新虚拟节点在关键节点内，进行替换
     * @param v
     * @param keyNode
     * @param candidatePipes
     * @param virtualNodeStList
     */
    public void updateV2Key(VirtualNodeSt v, InsertSt keyNode, List<KeyPipeSt> candidatePipes, List<VirtualNodeSt> virtualNodeStList) {
        // 对管道替换
        for (KeyPipeSt keyPipeSt: candidatePipes) {
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
     * @param virtualNodeStList
     */
    public void markFinalNodes(List<VirtualNodeSt> virtualNodeStList) {
        for (VirtualNodeSt virtualNodeSt: virtualNodeStList) {
            virtualNodeSt.setFinished(1);
        }
    }
}
