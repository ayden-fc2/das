package com.dwg.handler.dao;

import com.dwg.handler.entity.UploadDwgSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface UploadDwgStMapper {

    // 插入数据
    @Insert("INSERT INTO upload_dwg_st (project_name, user_id, dwg_path, json_path, is_public, created_time, is_deleted) " +
            "VALUES (#{projectName}, #{userId}, #{dwgPath}, #{jsonPath}, #{isPublic}, CURRENT_TIMESTAMP, 0)")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    boolean insert(UploadDwgSt uploadDwgSt);

    // 更新数据
    @Update("UPDATE upload_dwg_st SET project_name = #{projectName}, user_id = #{userId}, dwg_path = #{dwgPath}, " +
            "json_path = #{jsonPath}, is_public = #{isPublic}, created_time = #{createdTime} WHERE id = #{id}")
    int updateById(UploadDwgSt uploadDwgSt);

    // 根据 ID 删除数据
    @Delete("UPDATE upload_dwg_st SET is_deleted = 1 WHERE id = #{id}")
    int deleteById(@Param("id") int id);

    @Select("SELECT * FROM upload_dwg_st WHERE is_public = 1 AND is_deleted = 0")
    @Results({
            @Result(property = "projectName", column = "project_name"),
            @Result(property = "userId", column = "user_id"),
            @Result(property = "dwgPath", column = "dwg_path"),
            @Result(property = "jsonPath", column = "json_path"),
            @Result(property = "isPublic", column = "is_public"),
            @Result(property = "createdTime", column = "created_time")
    })
    List<UploadDwgSt> getPublicList();
}
