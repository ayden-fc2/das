package com.auth.oauth2.service;

import com.alibaba.fastjson.JSONObject;
import com.example.common.dto.ResponseBean;

import java.util.List;

public interface OrgService {
    String createOrg(int userId, String orgName, String orgDesc);

    boolean updateOrg(int userId, int orgId, String orgName, String orgDesc);

    String updateOrgCode(int userId, int orgId);

    boolean deleteOrg(int userId, int orgId);

    boolean joinOrg(int userId, String orgCode);

    boolean quitOrg(int userId, int orgId);

    List<JSONObject> getOrgsMember(int userId, int orgId);

    List<JSONObject> getMyOrgs(int userId, int page, int size);

    boolean deleteUser(int managerId, int orgId, int userId);

    boolean manageUserRoles(int managerId, int orgId, int userId, List<Integer> roleIds);

    Integer getMyOrgsNum(int userId);

    List<JSONObject> getAllMyOrgs(int userId);
}
