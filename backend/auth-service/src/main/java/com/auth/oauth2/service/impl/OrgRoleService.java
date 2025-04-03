package com.auth.oauth2.service.impl;

import org.springframework.security.core.Authentication;
import org.springframework.security.jwt.Jwt;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class OrgRoleService {

    /**
     * 检查当前用户是否在指定 orgId 下拥有 SUPER_MANAGER_TYPE 角色
     *
     * @param authentication 当前的认证信息
     * @param orgId 请求中传入的组织ID
     * @return true 表示校验通过，false 表示无权限
     */
    public boolean hasSuperManagerRole(Authentication authentication, String orgId) {
        System.out.println(orgId);
        return false;
    }
}
