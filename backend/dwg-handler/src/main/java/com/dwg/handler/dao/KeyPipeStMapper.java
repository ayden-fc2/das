package com.dwg.handler.dao;

import com.dwg.handler.entity.KeyPipeSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

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

    @Select("SELECT * FROM key_pipe_st WHERE dwg_id = #{projectId}")
    @Results({
            @Result(property = "keyPipeId", column = "key_pipe_id"),
            @Result(property = "startX", column = "start_x"),
            @Result(property = "startY", column = "start_y"),
            @Result(property = "endX", column = "end_x"),
            @Result(property = "endY", column = "end_y"),
            @Result(property = "dwgId", column = "dwg_id"),
            @Result(property = "startKeyHandle0", column = "start_key_handle0"),
            @Result(property = "startKeyHandle1", column = "start_key_handle1"),
            @Result(property = "endKeyHandle0", column = "end_key_handle0"),
            @Result(property = "endKeyHandle1", column = "end_key_handle1"),
            @Result(property = "vStartUUID", column = "v_start_uuid"),
            @Result(property = "vEndUUID", column = "v_end_uuid")
    })
    List<KeyPipeSt> selectByDwgId(long projectId);
}
