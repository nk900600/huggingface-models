const botImage =
  "https://liu.se/dfsmedia/dd35e243dfb7406993c1815aaf88a675/68881-50065/ai-header-1280";
export const userList = (data, isUser) => {
  let d = {
    id: Math.random(),
    description: data,
    src: isUser
      ? `https://xsgames.co/randomusers/avatar.php?g=pixel&key=1`
      : botImage,
  };
  return d
};
