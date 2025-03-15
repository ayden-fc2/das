package com.dwg.handler.dao;

import com.dwg.handler.entity.UploadDwgSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface UploadDwgStMapper {

    // 插入数据
    @Insert("INSERT INTO upload_dwg_st (project_name, user_id, dwg_path, json_path, is_public, created_time, is_deleted, analysised) " +
            "VALUES (#{projectName}, #{userId}, #{dwgPath}, #{jsonPath}, #{isPublic}, CURRENT_TIMESTAMP, 0, 0)")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    boolean insert(UploadDwgSt uploadDwgSt);

    // 更新数据
    @Update("UPDATE upload_dwg_st SET project_name = #{projectName}, user_id = #{userId}, dwg_path = #{dwgPath}, " +
            "json_path = #{jsonPath}, is_public = #{isPublic}, created_time = #{createdTime}, anaysised = #{analysised} WHERE id = #{id}")
    int updateById(UploadDwgSt uploadDwgSt);

    // 根据 ID 删除数据
    @Update("UPDATE upload_dwg_st SET is_deleted = 1 WHERE id = #{id}")
    int deleteById(@Param("id") int id);

    // 查询公共数据
    @Select("SELECT * FROM upload_dwg_st WHERE is_public = 1 AND is_deleted = 0")
    @Results({
            @Result(property = "projectName", column = "project_name"),
            @Result(property = "userId", column = "user_id"),
            @Result(property = "dwgPath", column = "dwg_path"),
            @Result(property = "jsonPath", column = "json_path"),
            @Result(property = "isPublic", column = "is_public"),
            @Result(property = "createdTime", column = "created_time"),
            @Result(property = "analysised", column = "analysised")
    })
    List<UploadDwgSt> getPublicList();

    // 设置一条数据为已分析
    @Update("UPDATE upload_dwg_st SET analysised = 1 WHERE id = #{id}")
    boolean setAnalysed(@Param("id") long id);

    @Select("SELECT analysised FROM upload_dwg_st WHERE id = #{dwgId}")
    boolean isAnalysed(Long dwgId);
}
