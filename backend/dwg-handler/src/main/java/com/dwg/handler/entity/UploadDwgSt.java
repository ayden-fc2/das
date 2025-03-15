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
    private Integer analysised;

    // Getter 和 Setter 方法

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getProjectName() {
        return projectName;
    }

    public void setProjectName(String projectName) {
        this.projectName = projectName;
    }

    public Integer getUserId() {
        return userId;
    }

    public void setUserId(Integer userId) {
        this.userId = userId;
    }

    public String getDwgPath() {
        return dwgPath;
    }

    public void setDwgPath(String dwgPath) {
        this.dwgPath = dwgPath;
    }

    public String getJsonPath() {
        return jsonPath;
    }

    public void setJsonPath(String jsonPath) {
        this.jsonPath = jsonPath;
    }

    public Integer getIsPublic() {
        return isPublic;
    }

    public void setIsPublic(Integer isPublic) {
        this.isPublic = isPublic;
    }

    public LocalDateTime getCreatedTime() {
        return createdTime;
    }

    public void setCreatedTime(LocalDateTime createdTime) {
        this.createdTime = createdTime;
    }

    public Integer getAnalysised() {
        return analysised;
    }

    public void setAnalysised(Integer analysised) {
        this.analysised = analysised;
    }
}
