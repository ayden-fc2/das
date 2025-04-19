package com.auth.oauth2.controller;

import com.alibaba.fastjson.JSONObject;
import com.auth.oauth2.service.OrgService;
import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.example.common.service.MyTokenService;
import feign.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/org")
public class OrgController {
    @Autowired
    MyTokenService myTokenService;

    @Autowired
    OrgService orgService;

    // 用户创建一个新群组，返回组织代码，给用户分配所有角色
    @GetMapping("/create")
    public ResponseBean<String> createOrg(
            @RequestHeader("Authorization") String token,
            @Param("orgName") String orgName,
            @Param("orgDesc") String orgDesc
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.createOrg(userId, orgName, orgDesc));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }
    // 管理员修改群组信息
    @GetMapping("/update")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType()) " +
            ")")
    public ResponseBean<Boolean> updateOrg(
            @RequestHeader("Authorization") String token,
            @Param("orgId") int orgId,
            @Param("orgName") String orgName,
            @Param("orgDesc") String orgDesc
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.updateOrg(userId, orgId, orgName, orgDesc));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 管理员更新群组代码
    @GetMapping("/updateCode")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<String> updateOrgCode(
            @RequestHeader("Authorization") String token,
            @Param("orgId") int orgId
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.updateOrgCode(userId, orgId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 管理员删除群组
    @GetMapping("/delete")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<Boolean> deleteOrg(
            @RequestHeader("Authorization") String token,
            @Param("orgId") int orgId
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.deleteOrg(userId, orgId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 用户通过组织代码加入群组 TODO: 重复加入
    @GetMapping("/join")
    public ResponseBean<Boolean> joinOrg(
            @RequestHeader("Authorization") String token,
            @Param("orgCode") String orgCode
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.joinOrg(userId, orgCode));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 用户退出群组
    @GetMapping("/quit")
    public ResponseBean<Boolean> quitOrg(
            @RequestHeader("Authorization") String token,
            @Param("orgId") int orgId
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.quitOrg(userId, orgId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 管理员分页获取群组用户信息
    @GetMapping("/getOrgsMember")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<List<JSONObject>> getOrgsMember(
            @RequestHeader("Authorization") String token,
            @Param("orgId") int orgId,
            @Param("page") int page,
            @Param("size") int size
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.getOrgsMember(userId, orgId, page, size));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 用户获取所有与自己相关的群组列表(分页查询)
    @GetMapping("/getMyOrgs")
    public ResponseBean<List<JSONObject>> getMyOrgs(
            @RequestHeader("Authorization") String token,
            @Param("page") int page,
            @Param("size") int size
    ) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.getMyOrgs(userId, page, size));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 用户获取自己加入的群组列表页数
    @GetMapping("/getMyOrgsNum")
    public ResponseBean<Integer> getMyOrgsPageNum(@RequestHeader("Authorization") String token) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.getMyOrgsNum(userId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 管理员删除用户
    @GetMapping("/deleteUser")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<Boolean> deleteUser(
            @RequestHeader("Authorization") String token,
            @Param("orgId") int orgId,
            @Param("userId") int userId
    ) {
        try {
            int managerId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(orgService.deleteUser(managerId, orgId, userId));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 管理员管理用户角色
    @GetMapping("/manageUserRoles")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Manager.getType())" +
            ")")
    public ResponseBean<Boolean> manageUserRoles(
            @RequestHeader("Authorization") String token,
            @Param("orgId") int orgId,
            @Param("userId") int userId,
            @Param("roleIds") String roleIds
    ) {
        try {
            int managerId = myTokenService.tokenToUserId(token);
            List<Integer> roleIdList = Arrays.stream(roleIds.split(","))
                                            .map(String::trim)       // Remove any extra spaces
                                            .map(Integer::parseInt)  // Convert to Integer
                                            .collect(Collectors.toList());
            return ResponseBean.success(orgService.manageUserRoles(managerId, orgId, userId, roleIdList));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }
}
