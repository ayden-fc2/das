package com.dwg.handler.dao;

import com.alibaba.fastjson.JSONObject;
import com.dwg.handler.entity.ProjectSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface ProjectStMapper {


    @Insert("INSERT INTO project_st (created_time, title, `desc`, parent_key, upload_id, org_id, creater_id) VALUES (CURRENT_TIMESTAMP, #{title}, #{description}, -1, -1, #{orgId}, #{createrId})")
    Boolean insertProject(@Param("orgId") String orgId, @Param("title") String title, @Param("description") String description, @Param("createrId") int createrId);

    @Select(
            "SELECT " +
                    "p.project_key, " +
                    "p.created_time, " +
                    "p.title, " +
                    "`p`.`desc`, " +  // 使用反引号防止关键字冲突
                    "p.parent_key, " +
                    "p.children_key, " +
                    "p.upload_id, " +
                    "p.org_id, " +
                    "p.creater_id, " +
                    "a.phoneNum, " +
                    "a.nickName, " +
                    "d.dwg_path, " +
                    "d.json_path, " +
                    "d.analysised " +
                    "FROM project_st p " +
                    "LEFT JOIN account_st a ON p.creater_id = a.accountId " +
                    "LEFT JOIN upload_dwg_st d ON p.upload_id = d.id " +
                    "WHERE p.org_id = #{orgId}"
    )
    @Results({
            @Result(property = "projectKey", column = "project_key"),
            @Result(property = "createdTime", column = "created_time"),
            @Result(property = "title", column = "title"),
            @Result(property = "desc", column = "desc"),
            @Result(property = "parentKey", column = "parent_key"),
            @Result(property = "childrenKey", column = "children_key"),
            @Result(property = "uploadId", column = "upload_id"),
            @Result(property = "orgId", column = "org_id"),
            @Result(property = "createrId", column = "creater_id"),
            @Result(property = "createrPhoneNum", column = "phoneNum"),
            @Result(property = "createrNickName", column = "nickName"),
            @Result(property = "dwgPath", column = "dwg_path"),
            @Result(property = "jsonPath", column = "json_path"),
            @Result(property = "analysised", column = "analysised")
    })
    List<JSONObject> selectProjectsByOrgId(@Param("orgId") String orgId);


    @Insert("INSERT INTO project_st (created_time, title, `desc`, parent_key, upload_id, org_id, creater_id) VALUES (CURRENT_TIMESTAMP, #{title}, #{description}, #{projectKey}, -1, #{orgId}, #{createrId})")
    Boolean insertChildProject( @Param("orgId") String orgId, @Param("projectKey") long projectKey, @Param("title") String title, @Param("description") String description, @Param("createrId") int createrId);

    @Delete("DELETE FROM project_st WHERE project_key = #{projectKey}")
    Boolean deleteProject(@Param("projectKey") long projectKey);

    @Update("UPDATE project_st SET upload_id = #{uploadId} WHERE project_key = #{projectId}")
    Boolean updateProjectUploadId( @Param("projectId") Long projectId, @Param("uploadId") long uploadId);
}
