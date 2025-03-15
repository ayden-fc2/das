package com.dwg.handler.service;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

public interface DwgService {
    Boolean genAnalysis(int userId, String projectName, String dwgPath, int isPublic);

    boolean isAnalysed(long projectId);

    boolean genGrahpML(int userId, JSONArray blockData, JSONArray insertsData, JSONArray pipesData, Long projectId);
}
