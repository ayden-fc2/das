package com.dwg.handler.service;

import com.alibaba.fastjson.JSONArray;
import com.dwg.handler.entity.UploadDwgSt;

import java.util.List;

public interface ReadService {
    List<UploadDwgSt> getPublicList();

    JSONArray getProjectGraph(long projectId);
}
