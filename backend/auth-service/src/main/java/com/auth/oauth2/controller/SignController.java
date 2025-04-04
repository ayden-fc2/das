package com.auth.oauth2.controller;

import com.auth.oauth2.service.SignService;
import com.example.common.dto.ResponseBean;
import com.example.common.service.MyTokenService;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/sign")
public class SignController {

    @Autowired
    SignService ss;

    //授权
    @GetMapping("/test")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).SuperManager.getType(), " +
            "T(com.example.common.enums.UserType).Manager.getType(), " +
            "T(com.example.common.enums.UserType).Controller.getType(), " +
            "T(com.example.common.enums.UserType).Observer.getType())" +
            ")")
    public ResponseBean<String> testFuc(@RequestHeader("Authorization") String token,@Param("orgId") Integer orgId){return ResponseBean.success("hello");}

    //登录
    @GetMapping("/signInCheck")
    public ResponseBean<Integer> signInCheck(@RequestParam("email") String phoneNum, @RequestParam("password") String password){return ss.signInCheck(phoneNum, password);}

    //发送验证码
    @GetMapping("/getPhoneCode")
    public ResponseBean<Integer> getPhoneCode(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode){return ss.getPhoneCode(phoneNum, mode);}

    //注册
    @GetMapping("/signUp")
    public ResponseBean<Integer> signIn(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("password") String password, @RequestParam("name") String name){return ss.signUp(phoneNum, mode, code, password, name);}

    //修改密码
    @GetMapping("/resetPassword")
    public ResponseBean<Integer> resetPassword(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("newPassword") String newPassword){return ss.resetPassword(phoneNum, mode, code, newPassword);}

}
