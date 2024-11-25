package com.dwg.handler.entity;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class UploadDwgSt {

    private Integer id;
    private String projectName;
    private Integer userId;
    private String dwgPath;
    private String jsonPath;
    private Integer isPublic;
    private LocalDateTime createdTime;
}
