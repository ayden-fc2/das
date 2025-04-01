package com.auth.oauth2.mapper;

import com.auth.oauth2.entity.AccountSt;
import com.auth.oauth2.entity.RelationshipSt;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface UserMapper {

    @Select("SELECT * FROM account_st WHERE phoneNum = #{phoneNum}")
    List<AccountSt> selectAccounts(@Param("phoneNum") String phoneNum);


    @Select("SELECT * FROM relationship_st WHERE accountId = #{userId}")
    List<RelationshipSt> selectRelations(long userId);

    @Insert("INSERT INTO account_st(phoneNum, passwordDetail, nickName) VALUES(#{phoneNum}, #{passwordDetail}, #{nickName})")
    @Options(useGeneratedKeys = true, keyProperty = "accountId", keyColumn = "accountId")
    void insertNewUser(AccountSt accountSt);

    @Insert("INSERT INTO relationship_st(accountId, authorityId) VALUES(#{accountId}, #{authorityId})")
    void insertRelationship(@Param("accountId") long accountId, @Param("authorityId") long authorityId);

    @Update("UPDATE account_st SET passwordDetail = #{newPassword} WHERE phoneNum = #{phoneNum}")
    void updatePassword(@Param("phoneNum") String phoneNum, @Param("newPassword") String newPassword);

    @Select("SELECT nickName FROM account_st WHERE accountId = #{userId}")
    String selectNicknameByUserId(@Param("userId") int userId);
}
