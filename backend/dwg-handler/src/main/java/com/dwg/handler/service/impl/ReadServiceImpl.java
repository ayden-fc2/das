package com.dwg.handler.service.impl;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.dao.BlockStMapper;
import com.dwg.handler.dao.InsertStMapper;
import com.dwg.handler.dao.UploadDwgStMapper;
import com.dwg.handler.entity.BlockSt;
import com.dwg.handler.entity.InsertSt;
import com.dwg.handler.entity.UploadDwgSt;
import com.dwg.handler.service.ReadService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

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
}
