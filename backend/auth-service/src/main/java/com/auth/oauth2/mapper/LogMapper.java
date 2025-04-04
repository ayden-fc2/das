package com.auth.oauth2.mapper;

import com.auth.oauth2.entity.LogSt;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface LogMapper {
    @Insert("INSERT INTO log_st(userId, copDetail, copType, copTime) VALUES(#{userId}, #{copDetail}, #{copType}, CURRENT_TIMESTAMP)")
    Boolean addLog(int userId, String copDetail, int copType);

    @Select("SELECT * FROM log_st WHERE userId = #{userId} AND copTime BETWEEN #{startDate} AND #{endDate} ORDER BY copTime DESC")
    List<LogSt> queryLog(int userId, String startDate, String endDate);
}
