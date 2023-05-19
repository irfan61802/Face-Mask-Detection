$(document).ready(function() {
    $('#submit').click(function(e){  
        e.preventDefault();
        var name = $('#fname').val()+" "+$('#lname').val();
        $.ajax({
        url: "/upload_face",
        type: "get",
        data: {name: name},
        success: function(response) {
        $("#result").html(response.html);
       },
      });
    });
 });