1. Normal.png：
    法线贴图（凹凸细节）→ role=normal
2. AmbientOcclusion.png：
    环境遮蔽（缝隙更暗）→ role=occlusion
3. Roughness.png：
    粗糙度 → role=roughness
4. Metallic.png：
    金属度 → role=metallic
5. Opacity.png：
    透明度 → role=opacity
6. Height.png：
    高度/置换（MuJoCo 的标准 layer role 里没有 height；一般要么忽略，要么先烘焙成 normal）
7. Scan1.png：
    这通常是扫描导出的“备用颜色图/合成图”，不确定是否包含光照或 AO；如果你发现 BaseColor 太“平”，可以试试把 Scan1 当作 rgb 贴图，但规范上仍是 BaseColor。