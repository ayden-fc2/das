package com.dwg.handler.controller;

import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.entity.InsertSt;
import com.dwg.handler.service.DwgService;
import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.example.common.service.MyTokenService;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/cop")
public class DwgController {
    @Autowired
    MyTokenService tokenService;

    @Autowired
    DwgService dwgService;

    // 管理员demo 上传文件后分析并存储，分析存储完毕后存表返回boolean
    @GetMapping("/genAnalysis")
    @PreAuthorize("@orgRoleService.isSuperManager(#token)")
    public ResponseBean<Long> genAnalysis(@RequestHeader("Authorization") String token,
                                          @Param("projectName") String projectName,
                                          @Param("dwgPath") String dwgPath,
                                          @Param("isPublic") int isPublic) {
        try {
            int userId = tokenService.tokenToUserId(token);
            return ResponseBean.success(dwgService.genAnalysis(userId, projectName, dwgPath, isPublic));
        } catch (Exception e) {
            throw new MyException(e.getMessage());
        }
    }

    // 项目内工程师上传解析
    @GetMapping("/uploadDwgByOrgId")
    @PreAuthorize("@orgRoleService.hasAnyRoleInOrg(" +
            "#token," +
            " #orgId," +
            " T(java.util.Arrays).asList(" +
            "T(com.example.common.enums.UserType).Controller.getType())" +
            ")")
    public ResponseBean<Boolean> uploadDwgByOrgId(
            @RequestHeader("Authorization") String token,
            @Param("dwgPath") String dwgPath,
            @Param("orgId") Long orgId,
            @Param("projectId") Long projectId
    ) {
        try {
            long userId = tokenService.tokenToUserId(token);
            return ResponseBean.success(dwgService.uploadDwgByOrgId(dwgPath, orgId, userId, projectId));
        } catch (Exception e) {
            throw new MyException("Something went wrong");
        }
    }

    // 管理员demo 二次解析 TODO 权限暂时全局
    @PostMapping("/genAnalysisOverview")
//    @PreAuthorize("@orgRoleService.isSuperManager(#token)")
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

    // 通用-对某个项目故障溯源
    @GetMapping("/genTrace")
    public ResponseBean<JSONObject> genTrace(@Param("projectId") Long projectId, @Param("faultIds") String faultIds) {
        try {
            return ResponseBean.success(dwgService.genTrace(projectId, faultIds));
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException("Something went wrong");
        }
    }


}
