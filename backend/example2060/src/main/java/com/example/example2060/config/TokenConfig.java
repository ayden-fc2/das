package com.example.example2060.config;

import com.example.common.enums.UserType;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.oauth2.common.DefaultOAuth2AccessToken;
import org.springframework.security.oauth2.common.OAuth2AccessToken;
import org.springframework.security.oauth2.provider.OAuth2Authentication;
import org.springframework.security.oauth2.provider.token.TokenEnhancer;
import org.springframework.security.oauth2.provider.token.TokenEnhancerChain;
import org.springframework.security.oauth2.provider.token.TokenStore;
import org.springframework.security.oauth2.provider.token.store.JwtAccessTokenConverter;
import org.springframework.security.oauth2.provider.token.store.JwtTokenStore;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

@Configuration
public class TokenConfig {
    @Bean
    public TokenStore jwtTokenStore() {
        return new JwtTokenStore(jwtTokenConverter());
    }

    @Bean
    protected JwtAccessTokenConverter jwtTokenConverter() {
        JwtAccessTokenConverter converter = new JwtAccessTokenConverter();
        converter.setSigningKey("springcloud123");
        return converter;
    }

    @Bean
    public TokenEnhancerChain tokenEnhancerChain() {
        TokenEnhancerChain tokenEnhancerChain = new TokenEnhancerChain();
        tokenEnhancerChain.setTokenEnhancers(Arrays.asList(customTokenEnhancer(), jwtTokenConverter()));
        return tokenEnhancerChain;
    }

    @Bean
    public TokenEnhancer customTokenEnhancer() {
        return new CustomTokenEnhancer();
    }

    private static class CustomTokenEnhancer implements TokenEnhancer {

        @Override
        public OAuth2AccessToken enhance(OAuth2AccessToken accessToken, OAuth2Authentication authentication) {

            // 获取原本的 authorities
            Collection<? extends GrantedAuthority> authorities = authentication.getAuthorities();

            // 将 authorities 转换为字符串数组
            String[] authorityArray = authorities.stream()
                    .map(GrantedAuthority::getAuthority)
                    .toArray(String[]::new);

            // 获取参数
            Map<String, String> requestParameters = authentication.getOAuth2Request().getRequestParameters();
            String isSuperManager = requestParameters.get("isSuperManager");
            String isManager = requestParameters.get("isManager");
            String isController = requestParameters.get("isController");
            String isObserver = requestParameters.get("isObserver");
            String userId = requestParameters.get("userId");


            // 根据参数删除不需要的权限
            if ("0".equals(isSuperManager)) {
                authorityArray = Arrays.stream(authorityArray)
                        .filter(authority -> !UserType.SuperManager.getType().equals(authority))
                        .toArray(String[]::new);
            }
            if ("0".equals(isManager)) {
                authorityArray = Arrays.stream(authorityArray)
                        .filter(authority -> !UserType.Manager.getType().equals(authority))
                        .toArray(String[]::new);
            }
            if ("0".equals(isController)) {
                authorityArray = Arrays.stream(authorityArray)
                        .filter(authority -> !UserType.Controller.getType().equals(authority))
                        .toArray(String[]::new);
            }
            if ("0".equals(isObserver)) {
                authorityArray = Arrays.stream(authorityArray)
                        .filter(authority -> !UserType.Observer.getType().equals(authority))
                        .toArray(String[]::new);
            }

            // 创建一个新的 token 对象
            DefaultOAuth2AccessToken updatedToken = new DefaultOAuth2AccessToken(accessToken);

            // 设置新的 authorities
            updatedToken.getAdditionalInformation().put("authorities", authorityArray);
            updatedToken.getAdditionalInformation().put("userId", userId);

            // 返回更新后的 token
            return updatedToken;
        }
    }

}

