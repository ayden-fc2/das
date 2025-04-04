package com.auth.oauth2.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.auth.oauth2.entity.OrgSt;
import com.auth.oauth2.mapper.OrgMapper;
import com.auth.oauth2.mapper.RelationshipMapper;
import com.auth.oauth2.service.OrgService;
import com.example.common.exception.MyException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Collections;
import java.util.List;
import java.util.UUID;

@Service
public class OrgServiceImpl implements OrgService {
    @Autowired
    OrgMapper orgMapper;

    @Autowired
    RelationshipMapper relationshipMapper;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public String createOrg(int userId, String orgName, String orgDesc) {
        // 创建org并返回随机的code
        String orgCode = UUID.randomUUID().toString().replace("-", "");
        OrgSt newOrg = new OrgSt();
        newOrg.setOrgCode(orgCode);
        newOrg.setOrgName(orgName);
        newOrg.setOrgDesc(orgDesc);
        newOrg.setCreaterId(userId);
        orgMapper.createOrg(newOrg);
        int orgId = (int) newOrg.getOrgId();
        for (int i = 0; i < 3; i++) {
            relationshipMapper.createRelationship(userId, orgId, i + 1);
        }
        // 创建用户（创建者）角色关系
        return orgCode;
    }

    @Override
    public boolean updateOrg(int userId, int orgId, String orgName, String orgDesc) {
        return orgMapper.updateOrg(userId, orgId, orgName, orgDesc);
    }

    @Override
    public String updateOrgCode(int userId, int orgId) {
        String newOrgCode = UUID.randomUUID().toString().replace("-", "");
        orgMapper.updateOrgCode(userId, orgId, newOrgCode);
        return newOrgCode;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean deleteOrg(int userId, int orgId) {
        return orgMapper.deleteOrg(userId, orgId) && relationshipMapper.deleteRelationshipByOrgId(orgId);
    }

    @Override
    public boolean joinOrg(int userId, String orgCode) {
        int orgId = orgMapper.getOrgIdByCode(orgCode);
        if (relationshipMapper.getRelationship(userId, orgId).isEmpty()) {
            throw new MyException("already joined！");
        }
        if (orgId > 0) {
            relationshipMapper.createRelationship(userId, orgId, 1);
            return true;
        }
        return false;
    }

    @Override
    public boolean quitOrg(int userId, int orgId) {
        return relationshipMapper.quitOrg(userId, orgId);
    }

    @Override
    public List<JSONObject> getOrgsMember(int userId, int orgId, int page, int size) {
        return relationshipMapper.getOrgsMember(orgId, (page - 1) * size, size);
    }

    @Override
    public List<JSONObject> getMyOrgs(int userId, int page, int size) {
        return relationshipMapper.getMyOrgs(userId, (page - 1) * size, size);
    }

    @Override
    public boolean deleteUser(int managerId, int orgId, int userId) {
        return relationshipMapper.deleteUser(managerId, orgId, userId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean manageUserRoles(int managerId, int orgId, int userId, List<Integer> roleIds) {
        relationshipMapper.deleteUser(managerId, orgId, userId);
        for (Integer roleId : roleIds) {
            relationshipMapper.createRelationship(userId, orgId, roleId);
        }
        return true;
    }

}
