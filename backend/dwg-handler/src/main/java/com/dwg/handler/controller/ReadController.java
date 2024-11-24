package com.dwg.handler.controller;

import com.example.common.dto.ResponseBean;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/read")
public class ReadController {
    //观察者用户测试
    @GetMapping("/userTest")
    public ResponseBean userTestFuc(){return ResponseBean.success("hello");}

}
