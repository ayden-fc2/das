package com.auth.oauth2.controller;

import com.alibaba.fastjson.JSONObject;
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
    public ResponseBean<String> testFuc(){return ResponseBean.success("hello");}

    //检查是否有控制者用户权限
    @GetMapping("/controller/test")
    public ResponseBean<String> deliverTest(){return ResponseBean.success("hello");}

    //检查是否有管理员权限
    @GetMapping("/manager/test")
    public ResponseBean<String> managerTest(){return ResponseBean.success("hello");}

    //检查是否有超级管理员权限
    @GetMapping("/superManager/test")
    public ResponseBean<String> superManagerTest(){return ResponseBean.success("hello");}

    //登录
    @GetMapping("/sign/signInCheck")
    public ResponseBean<Integer> signInCheck(@RequestParam("email") String phoneNum, @RequestParam("password") String password){return ss.signInCheck(phoneNum, password);}

    //发送验证码
    @GetMapping("/sign/getPhoneCode")
    public ResponseBean<Integer> getPhoneCode(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode){return ss.getPhoneCode(phoneNum, mode);}

    //注册
    @GetMapping("/sign/signUp")
    public ResponseBean<Integer> signIn(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("password") String password, @RequestParam("name") String name){return ss.signUp(phoneNum, mode, code, password, name);}

    //修改密码
    @GetMapping("/sign/resetPassword")
    public ResponseBean<Integer> resetPassword(@RequestParam("email") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("newPassword") String newPassword){return ss.resetPassword(phoneNum, mode, code, newPassword);}

    //获取用户id
    @GetMapping("/getUserId")
    public ResponseBean<Integer> getUserId(@RequestHeader("Authorization") String token){
        return ResponseBean.success(ts.tokenToUserId(token));
    }

    // 获取用户信息
    @GetMapping("/getRoles")
    public ResponseBean<JSONObject> getUserInfo(@RequestHeader("Authorization") String token){
        return ResponseBean.success(ts.tokenToRoles(token));
    }

}
