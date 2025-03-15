package com.dwg.handler.dao;

import com.dwg.handler.entity.KeyPipeSt;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface KeyPipeStMapper {

    @Delete("DELETE FROM key_pipe_st WHERE dwg_id = #{projectId}")
    void deleteByDwgId(Long projectId);

    @Insert("INSERT INTO key_pipe_st ( " +
            "start_x, start_y, end_x, end_y, dwg_id, " +
            "start_key_handle0, start_key_handle1,  " +
            "end_key_handle0, end_key_handle1,  " +
            "v_start_uuid, v_end_uuid " +
        ") VALUES (" +
            "#{startX}, #{startY}, #{endX}, #{endY}, #{dwgId}, " +
            "#{startKeyHandle0}, #{startKeyHandle1}, " +
            "#{endKeyHandle0}, #{endKeyHandle1}, " +
            "#{vStartUUID}, #{vEndUUID}" +
        ")"
    )
    boolean insertKeyPipeSt(KeyPipeSt keyPipe);
}
