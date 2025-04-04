package com.auth.oauth2.controller;

import com.auth.oauth2.service.RoleService;
import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.example.common.service.MyTokenService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/role")
public class RoleController {

    @Autowired
    MyTokenService mts;

    @Autowired
    RoleService roleService;

    // 获取用户信息
    @GetMapping("/getRoles")
    public ResponseBean<Map<String, List<String>>> getUserInfo(@RequestHeader("Authorization") String token){
        return ResponseBean.success(mts.tokenToClaims(token));
    }

    // 获取用户基础信息
    @GetMapping("/getUserInfo")
    public ResponseBean<Map<String, Object>> getUserBaseInfo(@RequestHeader("Authorization") String token){
        try {
            int userId = mts.tokenToUserId(token);
            return ResponseBean.success(roleService.getUserBaseInfo(userId));
        } catch (Exception e) {
            throw new MyException("Something went wrong");
        }
    }
}
