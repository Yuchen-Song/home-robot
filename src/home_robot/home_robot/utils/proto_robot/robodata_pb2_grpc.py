# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import robodata_pb2 as robodata__pb2


class RobotDataStub(object):
    """Interface exported by the server."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetHistory = channel.unary_stream(
            "/robodata.RobotData/GetHistory",
            request_serializer=robodata__pb2.RoboTensor.SerializeToString,
            response_deserializer=robodata__pb2.RobotSummary.FromString,
        )
        self.ReceiveRobotData = channel.stream_stream(
            "/robodata.RobotData/ReceiveRobotData",
            request_serializer=robodata__pb2.RobotSummary.SerializeToString,
            response_deserializer=robodata__pb2.RobotSummary.FromString,
        )
        self.PlanHighLevelAction = channel.stream_stream(
            "/robodata.RobotData/PlanHighLevelAction",
            request_serializer=robodata__pb2.RobotSummary.SerializeToString,
            response_deserializer=robodata__pb2.RobotSummary.FromString,
        )
        self.Chat = channel.unary_stream(
            "/robodata.RobotData/Chat",
            request_serializer=robodata__pb2.LLMInput.SerializeToString,
            response_deserializer=robodata__pb2.ChatMsg.FromString,
        )


class RobotDataServicer(object):
    """Interface exported by the server."""

    def GetHistory(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ReceiveRobotData(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PlanHighLevelAction(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Chat(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_RobotDataServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "GetHistory": grpc.unary_stream_rpc_method_handler(
            servicer.GetHistory,
            request_deserializer=robodata__pb2.RoboTensor.FromString,
            response_serializer=robodata__pb2.RobotSummary.SerializeToString,
        ),
        "ReceiveRobotData": grpc.stream_stream_rpc_method_handler(
            servicer.ReceiveRobotData,
            request_deserializer=robodata__pb2.RobotSummary.FromString,
            response_serializer=robodata__pb2.RobotSummary.SerializeToString,
        ),
        "PlanHighLevelAction": grpc.stream_stream_rpc_method_handler(
            servicer.PlanHighLevelAction,
            request_deserializer=robodata__pb2.RobotSummary.FromString,
            response_serializer=robodata__pb2.RobotSummary.SerializeToString,
        ),
        "Chat": grpc.unary_stream_rpc_method_handler(
            servicer.Chat,
            request_deserializer=robodata__pb2.LLMInput.FromString,
            response_serializer=robodata__pb2.ChatMsg.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "robodata.RobotData", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class RobotData(object):
    """Interface exported by the server."""

    @staticmethod
    def GetHistory(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/robodata.RobotData/GetHistory",
            robodata__pb2.RoboTensor.SerializeToString,
            robodata__pb2.RobotSummary.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ReceiveRobotData(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/robodata.RobotData/ReceiveRobotData",
            robodata__pb2.RobotSummary.SerializeToString,
            robodata__pb2.RobotSummary.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def PlanHighLevelAction(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/robodata.RobotData/PlanHighLevelAction",
            robodata__pb2.RobotSummary.SerializeToString,
            robodata__pb2.RobotSummary.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def Chat(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/robodata.RobotData/Chat",
            robodata__pb2.LLMInput.SerializeToString,
            robodata__pb2.ChatMsg.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
