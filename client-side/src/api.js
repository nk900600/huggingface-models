import { useCallback } from "react";

const baseUrl = "http://13.51.175.146:8888/";


export const postQuery = (query, useCallback, errorCallback) => {
  fetch(`${baseUrl}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      useCallback(data);
    })
    .catch((error) => {
      errorCallback(error);
    });
};
