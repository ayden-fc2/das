package com.example.example2060.controller;

import com.example.common.dto.ResponseBean;
import com.example.common.service.MyTokenService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
public class TestController {
    @Autowired
    MyTokenService ts;
    //所有人测试
    @RequestMapping("/allTest")
    public ResponseBean<String> allTestFuc(){
        return ResponseBean.success(
            "hello"
    );}

    //所有用户测试
    @RequestMapping("/userTest")
    public ResponseBean<String> userTestFuc(@RequestHeader("Authorization") String token){return ResponseBean.success("hello" + ts.tokenToUserId(token));}

}
