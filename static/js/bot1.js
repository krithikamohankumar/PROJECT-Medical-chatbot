function botResponse(rawText) {
    fetch("/get", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      // You can send data in the query string or in the body if needed
      // body: JSON.stringify({ msg: rawText })
    })
      .then(response => response.text())
      .then(data => {
        console.log(rawText);
        console.log(data);
        appendMessage(BOT_NAME, BOT_IMG, "left", data);
      })
      .catch(error => {
        console.error('Error fetching response:', error);
        // Handle errors here
      });
  }
  