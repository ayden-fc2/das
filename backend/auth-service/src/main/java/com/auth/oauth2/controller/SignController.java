package com.auth.oauth2.controller;

import com.auth.oauth2.service.SignService;
import com.auth.oauth2.service.TokenService;
import com.example.common.dto.ResponseBean;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
public class SignController {

    @Autowired
    SignService ss;

    @Autowired
    TokenService ts;

    //授权
    @GetMapping("/test")
    public ResponseBean testFuc(){return ResponseBean.success("hello");}

    //检查是否有控制者用户权限
    @GetMapping("/controller/test")
    public ResponseBean deliverTest(){return ResponseBean.success("hello");}

    //检查是否有管理员权限
    @GetMapping("/manager/test")
    public ResponseBean managerTest(){return ResponseBean.success("hello");}

    //检查是否有超级管理员权限
    @GetMapping("/superManager/test")
    public ResponseBean superManagerTest(){return ResponseBean.success("hello");}

    //登录
    @GetMapping("/sign/signInCheck")
    public ResponseBean signInCheck(@RequestParam("email") String phoneNum, @RequestParam("password") String password){return ss.signInCheck(phoneNum, password);}

    //发送验证码
    @GetMapping("/sign/getPhoneCode")
    public ResponseBean getPhoneCode(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode){return ss.getPhoneCode(phoneNum, mode);}

    //注册
    @GetMapping("/sign/signUp")
    public ResponseBean signIn(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("password") String password){return ss.signUp(phoneNum, mode, code, password);}

    //修改密码
    @GetMapping("/sign/resetPassword")
    public ResponseBean resetPassword(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("newPassword") String newPassword){return ss.resetPassword(phoneNum, mode, code, newPassword);}

    //获取用户id
    @GetMapping("/getUserId")
    public ResponseBean getUserId(@RequestHeader("Authorization") String token){
        return ResponseBean.success(ts.tokenToUserId(token));
    }

    // 获取用户信息
    @GetMapping("/getRoles")
    public ResponseBean getUserInfo(@RequestHeader("Authorization") String token){
        return ResponseBean.success(ts.tokenToRoles(token));
    }

}
