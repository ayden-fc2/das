package com.dwg.handler.dao;

import com.dwg.handler.entity.BlockSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface BlockStMapper {
    // 根据 ID 删除数据
    @Delete("DELETE FROM block_st WHERE dwg_id = #{id}")
    boolean deleteByDwgId(@Param("id") long id);

    // 新增一条记录
    @Insert("INSERT INTO block_st (block_name, block_count, svg, handle0, handle1, dwg_id) " +
            "VALUES (#{blockName}, #{blockCount}, #{svg}, #{handle0}, #{handle1}, #{dwgId})")
    int insertBlockSt(BlockSt blockSt);

    @Select("SELECT * FROM block_st WHERE dwg_id = #{projectId}")
    @Results({
            @Result(property = "blockId", column = "block_id"),
            @Result(property = "blockName", column = "block_name"),
            @Result(property = "blockCount", column = "block_count"),
            @Result(property = "svg", column = "svg"),
            @Result(property = "handle0", column = "handle0"),
            @Result(property = "handle1", column = "handle1"),
            @Result(property = "dwgId", column = "dwg_id")
    })
    List<BlockSt> getBlockStListByDwgId(@Param("projectId") long projectId);
}
