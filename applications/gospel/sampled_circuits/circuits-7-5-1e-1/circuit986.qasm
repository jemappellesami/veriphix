OPENQASM 2.0;
include "qelib1.inc";
qreg q987[7];
cx q987[4],q987[5];
cx q987[3],q987[4];
cx q987[3],q987[2];
cx q987[2],q987[1];
cx q987[0],q987[1];
