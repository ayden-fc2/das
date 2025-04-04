package com.auth.oauth2.mapper;

import com.auth.oauth2.entity.OrgSt;
import org.apache.ibatis.annotations.*;

@Mapper
public interface OrgMapper {
    @Insert("INSERT INTO org_st (org_name, created_time, creater_id, org_desc, org_code) VALUES (#{orgName}, CURRENT_TIMESTAMP, #{createrId}, #{orgDesc}, #{orgCode})")
    @Options(useGeneratedKeys = true, keyProperty = "orgId")
    boolean createOrg(OrgSt orgSt);

    @Update("UPDATE org_st SET org_name = #{orgName}, org_desc = #{orgDesc} WHERE org_id = #{orgId} AND creater_id = #{userId}")
    boolean updateOrg(@Param("userId") int userId, @Param("orgId") int orgId, @Param("orgName") String orgName, @Param("orgDesc") String orgDesc);

    @Update("UPDATE org_st SET org_code = #{newOrgCode} WHERE org_id = #{orgId} AND creater_id = #{userId}")
    void updateOrgCode( @Param("userId") int userId, @Param("orgId") int orgId, @Param("newOrgCode") String newOrgCode);

    @Delete("DELETE FROM org_st WHERE org_id = #{orgId} AND creater_id = #{userId}")
    boolean deleteOrg( @Param("userId") int userId, @Param("orgId") int orgId);

    @Select("SELECT org_id FROM org_st WHERE org_code = #{orgCode}")
    int getOrgIdByCode(String orgCode);


}
