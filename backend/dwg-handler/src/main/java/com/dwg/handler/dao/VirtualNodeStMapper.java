package com.dwg.handler.dao;

import com.dwg.handler.entity.VirtualNodeSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface VirtualNodeStMapper {

    @Delete("DELETE FROM virtual_node_st WHERE dwg_id = #{projectId}")
    void deleteByDwgId(Long projectId);

    @Insert("INSERT INTO virtual_node_st (x, y, uuid, dwg_id, finished) " +
            "VALUES (#{x}, #{y}, #{uuid}, #{dwgId}, #{finished})")
    boolean insertVirtualNodeSt(VirtualNodeSt v);

    @Select("SELECT * FROM virtual_node_st WHERE dwg_id = #{projectId}")
    @Results({
            @Result(property = "x", column = "x"),
            @Result(property = "y", column = "y"),
            @Result(property = "uuid", column = "uuid"),
            @Result(property = "dwgId", column = "dwg_id"),
            @Result(property = "finished", column = "finished"),
            @Result(property = "vNodeId", column = "v_node_id"),
    })
    List<VirtualNodeSt> selectByDwgId(Long projectId);
}
