package com.auth.oauth2.controller;

import com.auth.oauth2.entity.LogSt;
import com.auth.oauth2.service.LogService;
import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.example.common.service.MyTokenService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/logs")
public class LogController {

    @Autowired
    LogService ls;
    @Autowired
    MyTokenService myTokenService;

    // 新增一条日志
    @GetMapping("/add")
    public ResponseBean<Boolean> addLog(@RequestHeader("Authorization") String token, @RequestParam("copDetail") String copDetail, @RequestParam("copType") int copType) {
        // 业务逻辑
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(ls.addLog(userId, copDetail, copType));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 查询指定用户所有日志
    @GetMapping("/query")
    public ResponseBean<List<LogSt>> queryLog(@RequestHeader("Authorization") String token, @RequestParam("startDate") String startDate, @RequestParam("endDate") String endDate) {
        try {
            int userId = myTokenService.tokenToUserId(token);
            return ResponseBean.success(ls.queryLog(userId, startDate, endDate));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }
}
