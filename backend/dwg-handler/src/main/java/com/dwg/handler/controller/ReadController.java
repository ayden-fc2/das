package com.dwg.handler.controller;

import com.alibaba.fastjson.JSONArray;
import com.dwg.handler.entity.GraphDto;
import com.dwg.handler.entity.UploadDwgSt;
import com.dwg.handler.service.ReadService;
import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/read")
public class ReadController {
    @Autowired
    ReadService readService;

    // 观察者用户测试
    @GetMapping("/userTest")
    public ResponseBean userTestFuc(){return ResponseBean.success("hello");}

    //任意用户demo 获取公共demo列表
    @GetMapping("/getPublicList")
    public ResponseBean<List<UploadDwgSt>> getPublicList() {
        try {
            List<UploadDwgSt> result = readService.getPublicList();
            return ResponseBean.success(result);
        }catch (Exception e){
            throw new MyException(e.getMessage());
        }
    }

    //任意用户demo 获取project components分析结果
    @GetMapping("/getProjectGraph")
    public ResponseBean<JSONArray> getProjectGraph(@Param("projectId") long projectId) {
        try {
            JSONArray result = readService.getProjectGraph(projectId);
            return ResponseBean.success(result);
        }catch (Exception e){
            throw new MyException(e.getMessage());
        }
    }

    //任意用户demo 获取project 图结构
    @GetMapping("/getProjectGraphStructure")
    public ResponseBean<List<GraphDto>> getProjectGraphStructure(@Param("projectId") long projectId) {
        try {
            List<GraphDto> result = readService.getProjectGraphStructure(projectId);
            return ResponseBean.success(result);
        }catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }
}
