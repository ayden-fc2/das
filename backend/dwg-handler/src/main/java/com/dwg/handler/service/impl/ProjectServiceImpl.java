package com.dwg.handler.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.dao.ProjectStMapper;
import com.dwg.handler.entity.ProjectSt;
import com.dwg.handler.service.ProjectService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;

@Service
public class ProjectServiceImpl implements ProjectService {

    @Autowired
    ProjectStMapper pm;

    @Override
    public List<JSONObject> getProjectsByOrgId(String orgId) {
        return pm.selectProjectsByOrgId(orgId);
    }

    @Override
    public Boolean addProjectByOrgId(String orgId, String title, String description, int createrId) {
        return pm.insertProject(orgId, title, description, createrId);
    }

    @Override
    public Boolean addChildProjectByProjectKey(String orgId, String projectKey, String title, String description, int createrId) {
        return null;
    }

    @Override
    public Boolean updateProjectByProjectKey(String orgId, String title, String description, int createrId) {
        return null;
    }

    @Override
    public Boolean deleteProjectByProjectKey(String orgId, String projectKey, int createrId) {
        return null;
    }
}
