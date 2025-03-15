package com.dwg.handler.controller;

import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.service.DwgService;
import com.dwg.handler.service.TokenService;
import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.sun.org.apache.xpath.internal.operations.Bool;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/cop")
public class DwgController {
    @Autowired
    TokenService tokenService;

    @Autowired
    DwgService dwgService;

    // 上传文件后分析并存储，分析存储完毕后存表返回boolean
    @GetMapping("/genAnalysis")
    public ResponseBean<Boolean> genAnalysis(@RequestHeader("Authorization") String token,
                                          @Param("projectName") String projectName,
                                          @Param("dwgPath") String dwgPath,
                                          @Param("isPublic") int isPublic) {
        int userId = tokenService.tokenToUserId(token);
        System.out.println(userId);
        if (dwgService.genAnalysis(userId, projectName, dwgPath, isPublic)){
            return ResponseBean.success(true);
        }else {
            throw new MyException("Something went wrong");
        }
    }

    // 二次解析
    @PostMapping("/genAnalysisOverview")
    public ResponseBean<Boolean> genAnalysisOverview(@RequestHeader("Authorization") String token, @RequestBody JSONObject jsonObj) {
        int userId = tokenService.tokenToUserId(token);
        if (dwgService.isAnalysed(jsonObj.getLong("projectId"))) {
            return ResponseBean.success(true);
        } else if (dwgService.genGrahpML(
                userId,
                jsonObj.getJSONArray("blockData"),
                jsonObj.getJSONArray("insertsData"),
                jsonObj.getJSONArray("pipesData"),
                jsonObj.getLong("projectId")
        )) {
            return ResponseBean.success(true);
        } else {
            throw new MyException("Something went wrong");
        }
    }


}
