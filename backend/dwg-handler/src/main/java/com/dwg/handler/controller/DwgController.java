package com.dwg.handler.controller;

import com.example.common.dto.ResponseBean;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/cop")
public class DwgController {
    // 上传文件后分析并存储，先检查该用户上传数量不能超过10个，分析存储完毕后存表返回boolean
    @GetMapping("/genAnalysis")
    public ResponseBean genAnalysis() {
        return null;
    }

    // 获取公共demo列表
    @GetMapping("/getPublicList")
    public ResponseBean getPublicList() {
        return null;
    }
}
