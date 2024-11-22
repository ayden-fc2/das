package com.auth.oauth2.controller;

import com.auth.oauth2.service.SignService;
import com.example.common.dto.ResponseBean;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class SignController {

    @Autowired
    SignService ss;

    //授权
    @PostMapping("/test")
    public ResponseBean testFuc(){return ResponseBean.success("hello");}

    //检查是否有控制者用户权限
    @PostMapping("/controller/test")
    public ResponseBean deliverTest(){return ResponseBean.success("hello");}

    //检查是否有管理员权限
    @PostMapping("/manager/test")
    public ResponseBean managerTest(){return ResponseBean.success("hello");}

    //检查是否有超级管理员权限
    @PostMapping("/superManager/test")
    public ResponseBean superManagerTest(){return ResponseBean.success("hello");}

    //登录
    @PostMapping("/sign/signInCheck")
    public ResponseBean signInCheck(@RequestParam("phoneNum") String phoneNum, @RequestParam("password") String password){return ss.signInCheck(phoneNum, password);}

    //发送验证码
    @PostMapping("/sign/getPhoneCode")
    public ResponseBean getPhoneCode(@RequestParam("phoneNum") String phoneNum, @RequestParam("mode") int mode){return ss.getPhoneCode(phoneNum, mode);}

    //注册
    @PostMapping("/sign/signUp")
    public ResponseBean signIn(@RequestParam("phoneNum") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("password") String password){return ss.signUp(phoneNum, mode, code, password);}

    //修改密码
    @PostMapping("/sign/resetPassword")
    public ResponseBean resetPassword(@RequestParam("phoneNum") String phoneNum, @RequestParam("mode") int mode, @RequestParam("code") String code, @RequestParam("newPassword") String newPassword){return ss.resetPassword(phoneNum, mode, code, newPassword);}

    //获取用户id
    @PostMapping("/getUserId")
    public ResponseBean getUserId(@RequestHeader("Authorization") String token){
        return ss.getUserId(token);
    }

}
