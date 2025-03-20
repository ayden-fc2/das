package com.dwg.handler.service.impl;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.dao.*;
import com.dwg.handler.entity.*;
import com.dwg.handler.service.ReadService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
public class ReadServiceImpl implements ReadService {

    @Autowired
    UploadDwgStMapper uploadDwgStMapper;

    @Autowired
    BlockStMapper blockStMapper;

    @Autowired
    InsertStMapper insertStMapper;

    @Autowired
    VirtualNodeStMapper virtualNodeStMapper;

    @Autowired
    KeyPipeStMapper keyPipeStMapper;


    @Override
    public List<UploadDwgSt> getPublicList() {
        return uploadDwgStMapper.getPublicList();
    }

    @Override
    public JSONArray getProjectGraph(long projectId) {
        // 获取列表结果
        List<InsertSt> insertStList = insertStMapper.getInsertStListByDwgId(projectId);
        List<BlockSt> blockStList = blockStMapper.getBlockStListByDwgId(projectId);

        JSONArray result = new JSONArray();
        for (BlockSt blockSt: blockStList) {
            JSONObject blockObj = new JSONObject();
            JSONArray inserts = new JSONArray();
            for (InsertSt insertSt: insertStList) {
                if (insertSt.getBlockHandle0() == blockSt.getHandle0() && insertSt.getBlockHandle1() == blockSt.getHandle1()) {
                    JSONObject insertObj = new JSONObject();
                    insertObj.put("handle0", insertSt.getInsertHandle0());
                    insertObj.put("handle1", insertSt.getInsertHandle1());
                    insertObj.put("centerX", insertSt.getCenterPtX());
                    insertObj.put("centerY", insertSt.getCenterPtY());
                    insertObj.put("boxWidth", insertSt.getBoxWidth());
                    insertObj.put("boxHeight", insertSt.getBoxHeight());
                    insertObj.put("upstream", insertSt.getUpstream());
                    insertObj.put("downstream", insertSt.getDownstream());
                    inserts.add(insertObj);
                }
            }
            blockObj.put("inserts", inserts);
            blockObj.put("handle0", blockSt.getHandle0());
            blockObj.put("handle1", blockSt.getHandle1());
            blockObj.put("blockName", blockSt.getBlockName());
            blockObj.put("svg", blockSt.getSvg());
            result.add(blockObj);
        }
        return result;
    }

    /**
     * 获取结构图，将节点、边等铺平，type为key/virtual
     * @param projectId
     * @return
     */
    @Override
    public List<GraphDto> getProjectGraphStructure(long projectId) {
        List<GraphDto> graphDtoList = new ArrayList<>();
        // 获取所有真实节点连表，处理并添加进集合
        List<InsertSt> inserts = insertStMapper.getInsertStListByDwgId(projectId);
        List<BlockSt> blocks = blockStMapper.getBlockStListByDwgId(projectId);
        for (InsertSt insertSt: inserts) {
            BlockSt blockSt = blocks.stream().filter(b -> b.getHandle0() == insertSt.getBlockHandle0() && b.getHandle1() == insertSt.getBlockHandle1()).findFirst().orElse(null);
            GraphDto graphDto = new GraphDto();
            JSONObject position = new JSONObject();
            JSONObject box = new JSONObject();
            JSONObject stream = new JSONObject();
            position.put("x", insertSt.getCenterPtX());
            position.put("y", insertSt.getCenterPtY());
            box.put("width", insertSt.getBoxWidth());
            box.put("height", insertSt.getBoxHeight());
            stream.put("upstream", insertSt.getUpstream());
            stream.put("downstream", insertSt.getDownstream());
            graphDto.setId(insertSt.getInsertHandle0() + "-" + insertSt.getInsertHandle1());
            graphDto.setNode(blockSt);
            assert blockSt != null;
            graphDto.setLabel(blockSt.getBlockName());
            graphDto.setType("key");
            graphDto.setPosition(position);
            graphDto.setBox(box);
            graphDto.setStream(stream);
            graphDtoList.add(graphDto);
        }
        // 获取所有虚拟节点，处理并添加进集合
        List<VirtualNodeSt> vnodes = virtualNodeStMapper.selectByDwgId(projectId);
        for (VirtualNodeSt vnode: vnodes) {
            GraphDto graphDto = new GraphDto();
            JSONObject position = new JSONObject();
            position.put("x", vnode.getX());
            position.put("y", vnode.getY());
            graphDto.setId(vnode.getUuid());
            graphDto.setType("virtual");
            graphDto.setPosition(position);
            graphDtoList.add(graphDto);
        }
        // 获取所有边，处理并添加进集合
        List<KeyPipeSt> keyPipes = keyPipeStMapper.selectByDwgId(projectId);
        for (KeyPipeSt keyPipe: keyPipes) {
            GraphDto graphDto = new GraphDto();
            if (keyPipe.getVStartUUID() == null && keyPipe.getStartKeyHandle0() != -1 && keyPipe.getStartKeyHandle1() != -1) {
                // 起点为真实节点
                InsertSt sourceInsert = inserts.stream().filter(i -> i.getInsertHandle0() == keyPipe.getStartKeyHandle0() && i.getInsertHandle1() == keyPipe.getStartKeyHandle1()).findFirst().orElse(null);
                graphDto.setSource(sourceInsert.getInsertHandle0() + "-" + sourceInsert.getInsertHandle1());
            } else {
                // 起点为虚拟节点
                VirtualNodeSt sourceVnode = vnodes.stream().filter(v -> v.getUuid().equals(keyPipe.getVStartUUID())).findFirst().orElse(null);
                graphDto.setSource(sourceVnode.getUuid());
            }
            if (keyPipe.getVEndUUID() == null && keyPipe.getEndKeyHandle0() != -1 && keyPipe.getEndKeyHandle1() != -1) {
                // 终点为真实节点
                InsertSt targetInsert = inserts.stream().filter(i -> i.getInsertHandle0() == keyPipe.getEndKeyHandle0() && i.getInsertHandle1() == keyPipe.getEndKeyHandle1()).findFirst().orElse(null);
                graphDto.setTarget(targetInsert.getInsertHandle0() + "-" + targetInsert.getInsertHandle1());
            } else {
                // 终点为虚拟节点
                VirtualNodeSt targetVnode = vnodes.stream().filter(v -> v.getUuid().equals(keyPipe.getVEndUUID())).findFirst().orElse(null);
                graphDto.setTarget(targetVnode.getUuid());
            }
            graphDtoList.add(graphDto);
        }
        return graphDtoList;
    }
}
