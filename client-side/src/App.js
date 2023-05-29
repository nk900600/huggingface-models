import logo from "./logo.svg";
import "./App.css";
import { Input, Col, Row, Space, Card, Upload, Avatar, List } from "antd";
import {
  InboxOutlined,
  LoadingOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import { useState } from "react";
import {userList} from "./util"
import {postQuery} from "./api"
const { Dragger } = Upload;
const { Search } = Input;

const botImage =
  "https://liu.se/dfsmedia/dd35e243dfb7406993c1815aaf88a675/68881-50065/ai-header-1280";
const props = {
  action: "https://www.mocky.io/v2/5cc8019d300000980a055e76",
  onChange({ file, fileList }) {
    if (file.status !== "uploading") {
      //  file.status == "removed"
      console.log(file, fileList);
    }
  },
  defaultFileList: [
    {
      uid: "1",
      name: "Goku (anime series file)",
      status: "done",
      // url: "http://www.baidu.com/yyy.png",
    },
  ],
};
let data = [
  // {
  //   description:
  //     "Ant Design, a design language for background applications, is refined by Ant UED TeamAnt Design, a design language for background applications, is refined by Ant UED TeamAnt Design, a design language for background applications, is refined by Ant UED TeamAnt Design, a design language for background applications, is refined by Ant UED TeamAnt Design, a design language for background applications, is refined by Ant UED TeamAnt Design, a design language for background applications, is refined by Ant UED Team",
  //   src: `https://xsgames.co/randomusers/avatar.php?g=pixel&key=1`,
  // },
  // {
  //   description: "scscscsc",
  //   src: botImage,
  // },
];

function App() {
  const [loading, setLoading] = useState(false);
  const [chat, setChat] = useState(data);
  const [query, setQuery] = useState("");
  const [messagesEnd, setMessagesEnd] = useState(data);
  const handleClickSearch = (event) => {
 

    if (!event.currentTarget && query) {
      setLoading(true);
      postQuery(
        query,
        ({ ans, docs }) => {
          let da = userList(query, true);
          const newDatas = data.concat(da);
          data = newDatas;
          setChat(newDatas);

          let d = userList(
            `${ans}
          ------------Docs Refs--------------
          ${docs}
        `,
            false
          );
          const newData = data.concat(d);
          data = newData;
          setChat(newData);
          setQuery("");
          setTimeout(() => {
            scrollToBottom();
          });
          setLoading(false);
        },
        (error) => {
          setLoading(false);
        }
      );
    }
  };
  const handleChange = (event) => {
    setQuery(event.currentTarget.value);
  };
  const beforeUpload = (event) => {
    console.log(event.currentTarget.value);
  };

  const uploadButton = (
    <div>
      {loading ? <LoadingOutlined /> : <PlusOutlined />}
      <div style={{ marginTop: 8 }}>Upload</div>
    </div>
  );
  const scrollToBottom = () => {
    messagesEnd && messagesEnd.scrollIntoView({ behavior: "smooth" });
  };
  return (
    <div className="container">
      <div className="App">
        <Row>
          <Col span={6}>
            <div className="leftcontainer">
              <Card
                title="Documents used for Generative AI "
                style={{ width: 300 }}
              >
                <Upload {...props}></Upload>
                <div className="uploadeContainer">
                  {/* <Upload
                    name="avatar"
                    listType="picture-card"
                    className="avatar-uploader"
                    showUploadList={false}
                    action="https://www.mocky.io/v2/5cc8019d300000980a055e76"
                    beforeUpload={beforeUpload}
                    onChange={handleChange}
                  >
                    {uploadButton}
                  </Upload> */}
                </div>
              </Card>
            </div>
          </Col>
          <Col span={18}>
            <div>
              <div className="chat-container">
                <List
                  itemLayout="horizontal"
                  dataSource={chat}
                  renderItem={(item, index) => (
                    <>
                      <List.Item key="index">
                        <List.Item.Meta
                          avatar={<Avatar src={item.src} />}
                          description={item.description}
                        />
                      </List.Item>
                    </>
                  )}
                />
                <div
                  style={{ float: "left", clear: "both" }}
                  ref={(el) => {
                    setMessagesEnd(el);
                  }}
                ></div>
              </div>
              <Search
                onClick={handleClickSearch}
                onSearch={handleClickSearch}
                onChange={handleChange}
                placeholder="input search text"
                enterButton="Search"
                size="large"
                value={query}
                loading={loading}
              />
            </div>
          </Col>
        </Row>
      </div>
    </div>
  );
}

export default App;
