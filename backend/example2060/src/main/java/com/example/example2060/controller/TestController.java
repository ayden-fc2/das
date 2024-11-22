package com.example.example2060.controller;

import com.example.common.dto.ResponseBean;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {
    //所有人测试
    @RequestMapping("/allTest")
    public ResponseBean allTestFuc(){return ResponseBean.success("hello");}

    //所有用户测试
    @RequestMapping("/userTest")
    public ResponseBean userTestFuc(){return ResponseBean.success("hello");}

}
