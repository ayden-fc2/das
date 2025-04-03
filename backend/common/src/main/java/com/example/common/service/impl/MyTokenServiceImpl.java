package com.example.common.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.example.common.service.MyTokenService;
import org.springframework.security.jwt.Jwt;
import org.springframework.security.jwt.JwtHelper;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MyTokenServiceImpl implements MyTokenService {
    public int tokenToUserId(String token) {
        // 去掉 "Bearer " 前缀
        if (token.startsWith("Bearer ")) {
            token = token.substring(7);
        }

        Jwt jwt = JwtHelper.decode(token);
        String jsonString = jwt.getClaims();

        // 使用 FastJSON 解析 JSON 字符串
        JSONObject claimsJson = JSONObject.parseObject(jsonString);

        // 获取 "userId"
        return (Integer.parseInt(claimsJson.getString("userId")));
    }

    public Map<String, List<String>> tokenToClaims(String token) {
        // 去掉 "Bearer " 前缀
        if (token.startsWith("Bearer ")) {
            token = token.substring(7);
        }

        Jwt jwt = JwtHelper.decode(token);
        String jsonString = jwt.getClaims();

        // 使用 FastJSON 解析 JSON 字符串
        JSONObject claimsJson = JSONObject.parseObject(jsonString);

        // 获取 "orgRoles" 对象
        JSONObject orgRolesJson = claimsJson.getJSONObject("orgRoles");
        if (orgRolesJson == null) {
            return Collections.emptyMap();
        }

        // 将 JSONObject 转换为 Map<String, String>
        Map<String, List<String>> orgRolesMap = new HashMap<>();
        for (Map.Entry<String, Object> entry : orgRolesJson.entrySet()) {
            orgRolesMap.put(entry.getKey(), (List<String>) entry.getValue());
        }

        return orgRolesMap;
    }
}
