package com.dwg.handler.service;

import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.entity.ProjectSt;

import java.util.List;

public interface ProjectService {

    List<JSONObject> getProjectsByOrgId(String orgId);

    Boolean addProjectByOrgId(String orgId, String title, String description, int createrId);

    Boolean addChildProjectByProjectKey(String orgId, long projectKey, String title, String description, int createrId);

    Boolean updateProjectByProjectKey(String orgId, String title, String description, int createrId);

    Boolean deleteProjectByProjectKey(String orgId, long projectKey, int createrId);
}
