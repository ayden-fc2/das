package com.example.common.service.impl;

import com.example.common.enums.UserType;
import com.example.common.service.MyTokenService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class OrgRoleService {
    @Autowired
    MyTokenService mt;

    /**
     * 检查当前用户是否在指定 orgId 下角色
     */
    public boolean hasAnyRoleInOrg(String token, String orgId, List<String> roles) {
        Map<String, List<String>> claims = mt.tokenToClaims(token);

        // 获取该 orgId 下的角色列表
        List<String> userRoles = claims.get(orgId);
        if (userRoles == null || userRoles.isEmpty()) {
            return false; // 该 orgId 下没有任何角色
        }

        // 将用户角色转换为 Set 以加速查找
        Set<String> userRoleSet = new HashSet<>(userRoles);

        // 任何一个要求的角色在用户角色中存在即返回 true
        for (String role : roles) {
            if (userRoleSet.contains(role)) {
                return true;
            }
        }
        return false;
    }

    public boolean isSuperManager(String token) {
        Map<String, List<String>> claims = mt.tokenToClaims(token);
        return claims.values().stream()
                        .anyMatch(roleList -> roleList.contains(UserType.SuperManager.getType()));
    }
}
