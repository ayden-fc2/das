package com.dwg.handler.dao;

import com.dwg.handler.entity.VirtualNodeSt;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface VirtualNodeStMapper {

    @Delete("DELETE FROM virtual_node_st WHERE dwg_id = #{projectId}")
    void deleteByDwgId(Long projectId);

    @Insert("INSERT INTO virtual_node_st (x, y, uuid, dwg_id, finished) " +
            "VALUES (#{x}, #{y}, #{uuid}, #{dwgId}, #{finished})")
    boolean insertVirtualNodeSt(VirtualNodeSt v);
}
