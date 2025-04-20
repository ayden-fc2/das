package com.dwg.handler.controller;

import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.entity.ProjectSt;
import com.dwg.handler.service.ProjectService;
import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.example.common.service.MyTokenService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/project")
public class ProjectController {
    @Autowired
    ProjectService ps;

    @Autowired
    MyTokenService mts;

    /**
     * 获取用户所属组织的项目列表
     * @param token
     * @param orgId
     * @return
     */
    @RequestMapping("/getProjectsByOrgId")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Observer.getType())" +
            ")")
    public ResponseBean<List<JSONObject>> getProjectsByOrgId(
            @RequestHeader("Authorization") String token,
            @RequestParam("orgId") String orgId
    ) {
        try {
            return ResponseBean.success(ps.getProjectsByOrgId(orgId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    /**
     * 组织下新增项目
     * @param token
     * @param orgId
     * @param title
     * @param description
     * @return
     */
    @RequestMapping("/addProjectByOrgId")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<Boolean> addProjectByOrgId(
            @RequestHeader("Authorization") String token,
            @RequestParam("orgId") String orgId,
            @RequestParam("title") String title,
            @RequestParam("description") String description
    ) {
        try {
            int createrId = mts.tokenToUserId(token);
            return ResponseBean.success(ps.addProjectByOrgId(orgId, title, description, createrId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    /**
     * 项目下新增子项目
     * @param token
     * @param orgId
     * @param projectKey
     * @param title
     * @param description
     * @return
     */
    @RequestMapping("/addChildProjectByProjectKey")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<Boolean> addChildProjectByProjectKey(
            @RequestHeader("Authorization") String token,
            @RequestParam("orgId") String orgId,
            @RequestParam("projectKey") String projectKey,
            @RequestParam("title") String title,
            @RequestParam("description") String description
    ) {
        try {
            int createrId = mts.tokenToUserId(token);
            return ResponseBean.success(ps.addChildProjectByProjectKey(orgId, projectKey, title, description, createrId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    /**
     * 项目更新
     * @param token
     * @param orgId
     * @param title
     * @param description
     * @return
     */
    @RequestMapping("/updateProjectByProjectKey")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<Boolean> updateProjectByProjectKey(
            @RequestHeader("Authorization") String token,
            @RequestParam("orgId") String orgId,
            @RequestParam("title") String title,
            @RequestParam("description") String description
    ) {
        try {
            int createrId = mts.tokenToUserId(token);
            return ResponseBean.success(ps.updateProjectByProjectKey(orgId, title, description, createrId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }


    /**
     * 项目删除
     * @param token
     * @param orgId
     * @param projectKey
     * @return
     */
    @RequestMapping("/deleteProjectByProjectKey")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<Boolean> deleteProjectByProjectKey(
            @RequestHeader("Authorization") String token,
            @RequestParam("orgId") String orgId,
            @RequestParam("projectKey") String projectKey
    ) {
        try {
            int createrId = mts.tokenToUserId(token);
            return ResponseBean.success(ps.deleteProjectByProjectKey(orgId, projectKey, createrId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }


}
