package com.dwg.handler.dao;

import com.dwg.handler.entity.InsertSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface InsertStMapper {
    // 根据 ID 删除数据
    @Delete("DELETE FROM insert_st WHERE dwg_id = #{id}")
    boolean deleteByDwgId(@Param("id") long id);

    // 插入一条数据
    @Insert("INSERT INTO insert_st (block_handle0, block_handle1, insert_handle0, " +
            "insert_handle1, box_width, box_height, center_pt_x, center_pt_y, dwg_id) " +
            "VALUES (#{insert.blockHandle0}, #{insert.blockHandle1}, " +
            "#{insert.insertHandle0}, #{insert.insertHandle1}, #{insert.boxWidth}, " +
            "#{insert.boxHeight}, #{insert.centerPtX}, #{insert.centerPtY}, #{insert.dwgId})")
    int insertInsertSt(@Param("insert") InsertSt insertSt);

    @Select("SELECT * FROM insert_st WHERE dwg_id = #{projectId}")
    @Results({
            @Result(property = "insertId", column = "insert_id"),
            @Result(property = "blockHandle0", column = "block_handle0"),
            @Result(property = "blockHandle1", column = "block_handle1"),
            @Result(property = "insertHandle0", column = "insert_handle0"),
            @Result(property = "insertHandle1", column = "insert_handle1"),
            @Result(property = "boxWidth", column = "box_width"),
            @Result(property = "boxHeight", column = "box_height"),
            @Result(property = "centerPtX", column = "center_pt_x"),
            @Result(property = "centerPtY", column = "center_pt_y"),
            @Result(property = "dwgId", column = "dwg_id")
    })
    List<InsertSt> getInsertStListByDwgId(@Param("projectId") long projectId);
}
