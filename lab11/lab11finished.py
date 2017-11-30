



<!DOCTYPE html>
<html lang="en-US" >

<head>

	
	<script>
window.ts_endpoint_url = "https:\/\/slack.com\/beacon\/timing";

(function(e) {
	var n=Date.now?Date.now():+new Date,r=e.performance||{},t=[],a={},i=function(e,n){for(var r=0,a=t.length,i=[];a>r;r++)t[r][e]==n&&i.push(t[r]);return i},o=function(e,n){for(var r,a=t.length;a--;)r=t[a],r.entryType!=e||void 0!==n&&r.name!=n||t.splice(a,1)};r.now||(r.now=r.webkitNow||r.mozNow||r.msNow||function(){return(Date.now?Date.now():+new Date)-n}),r.mark||(r.mark=r.webkitMark||function(e){var n={name:e,entryType:"mark",startTime:r.now(),duration:0};t.push(n),a[e]=n}),r.measure||(r.measure=r.webkitMeasure||function(e,n,r){n=a[n].startTime,r=a[r].startTime,t.push({name:e,entryType:"measure",startTime:n,duration:r-n})}),r.getEntriesByType||(r.getEntriesByType=r.webkitGetEntriesByType||function(e){return i("entryType",e)}),r.getEntriesByName||(r.getEntriesByName=r.webkitGetEntriesByName||function(e){return i("name",e)}),r.clearMarks||(r.clearMarks=r.webkitClearMarks||function(e){o("mark",e)}),r.clearMeasures||(r.clearMeasures=r.webkitClearMeasures||function(e){o("measure",e)}),e.performance=r,"function"==typeof define&&(define.amd||define.ajs)&&define("performance",[],function(){return r}) // eslint-disable-line
})(window);

</script>
<script>


;(function() {



window.TSMark = function(mark_label) {
	if (!window.performance || !window.performance.mark) return;
	performance.mark(mark_label);
};
window.TSMark('start_load');


window.TSMeasureAndBeacon = function(measure_label, start_mark_label) {
	if (start_mark_label === 'start_nav' && window.performance && window.performance.timing) {
		window.TSBeacon(measure_label, (new Date()).getTime() - performance.timing.navigationStart);
		return;
	}
	if (!window.performance || !window.performance.mark || !window.performance.measure) return;
	performance.mark(start_mark_label + '_end');
	try {
		performance.measure(measure_label, start_mark_label, start_mark_label + '_end');
		window.TSBeacon(measure_label, performance.getEntriesByName(measure_label)[0].duration);
	} catch (e) {
		
	}
};


if ('sendBeacon' in navigator) {
	window.TSBeacon = function(label, value) {
		var endpoint_url = window.ts_endpoint_url || 'https://slack.com/beacon/timing';
		navigator.sendBeacon(endpoint_url + '?data=' + encodeURIComponent(label + ':' + value), '');
	};
} else {
	window.TSBeacon = function(label, value) {
		var endpoint_url = window.ts_endpoint_url || 'https://slack.com/beacon/timing';
		(new Image()).src = endpoint_url + '?data=' + encodeURIComponent(label + ':' + value);
	};
}
})();
</script>
 

<script>
window.TSMark('step_load');
</script>	<noscript><meta http-equiv="refresh" content="0; URL=/files/U71477WFQ/F88JBS1KR/lab11finished.py?nojsmode=1" /></noscript>
<script type="text/javascript">
if(self!==top)window.document.write("\u003Cstyle>body * {display:none !important;}\u003C\/style>\u003Ca href=\"#\" onclick="+
"\"top.location.href=window.location.href\" style=\"display:block !important;padding:10px\">Go to Slack.com\u003C\/a>");

(function() {
	var timer;
	if (self !== top) {
		timer = window.setInterval(function() {
			if (window.$) {
				try {
					$('#page').remove();
					$('#client-ui').remove();
					window.TS = null;
					window.clearInterval(timer);
				} catch(e) {}
			}
		}, 200);
	}
}());

</script>

<script>

(function() {


	window.callSlackAPIUnauthed = function(method, args, callback) {
		var timestamp = Date.now() / 1000;  
		var version = (window.TS && TS.boot_data && TS.boot_data.version_uid) ? TS.boot_data.version_uid.substring(0, 8) : 'noversion';
		var url = '/api/' + method + '?_x_id=' + version + '-' + timestamp;

		var req = new XMLHttpRequest();

		req.onreadystatechange = function() {
			if (req.readyState == 4) {
				req.onreadystatechange = null;
				var obj;

				if (req.status == 200 || req.status == 429) {
					try {
						obj = JSON.parse(req.responseText);
					} catch (err) {
						TS.warn(8675309, 'unable to do anything with api rsp');
					}
				}

				obj = obj || {
					ok: false,
				};

				callback(obj.ok, obj, args);
			}
		};

		var async = true;
		req.open('POST', url, async);

		var form_data = new FormData();
		var has_data = false;
		Object.keys(args).forEach(function(k) {
			if (k[0] === '_') return;
			form_data.append(k, args[k]);
			has_data = true;
		});

		if (has_data) {
			req.send(form_data);
		} else {
			req.send();
		}
	};
})();
</script>

	<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/webpack.manifest.b3b2c4c2b444fbb4e0c4.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

			
	
		<script>
			if (window.location.host == 'slack.com' && window.location.search.indexOf('story') < 0) {
				document.cookie = '__cvo_skip_doc=' + escape(document.URL) + '|' + escape(document.referrer) + ';path=/';
			}
		</script>
	

		<script type="text/javascript">
		
		try {
			if(window.location.hash && !window.location.hash.match(/^(#?[a-zA-Z0-9_]*)$/)) {
				window.location.hash = '';
			}
		} catch(e) {}
		
	</script>

	<script type="text/javascript">
				
			window.optimizely = [];
			window.dataLayer = [];
			window.ga = false;
		
	
				(function(e,c,b,f,d,g,a){e.SlackBeaconObject=d;
		e[d]=e[d]||function(){(e[d].q=e[d].q||[]).push([1*new Date(),arguments])};
		e[d].l=1*new Date();g=c.createElement(b);a=c.getElementsByTagName(b)[0];
		g.async=1;g.src=f;a.parentNode.insertBefore(g,a)
		})(window,document,"script","https://a.slack-edge.com/bv1-1/slack_beacon.5dbbc3dd9f37d8bc2f4e.min.js","sb");
		sb('set', 'token', '3307f436963e02d4f9eb85ce5159744c');

					sb('set', 'user_id', "U71EJ5FT9");
							sb('set', 'user_' + "batch", "signup_api");
							sb('set', 'user_' + "created", "2017-09-08");
						sb('set', 'name_tag', "byu-dl-f17" + '/' + "jaringson");
				sb('track', 'pageview');

		
		function track(a) {
			if(window.ga) ga('send','event','web', a);
			if(window.sb) sb('track', a);
		}
		

	</script>

	
	<meta name="referrer" content="no-referrer">
		<meta name="superfish" content="nofish">

	<script type="text/javascript">



var TS_last_log_date = null;
var TSMakeLogDate = function() {
	var date = new Date();

	var y = date.getFullYear();
	var mo = date.getMonth()+1;
	var d = date.getDate();

	var time = {
	  h: date.getHours(),
	  mi: date.getMinutes(),
	  s: date.getSeconds(),
	  ms: date.getMilliseconds()
	};

	Object.keys(time).map(function(moment, index) {
		if (moment == 'ms') {
			if (time[moment] < 10) {
				time[moment] = time[moment]+'00';
			} else if (time[moment] < 100) {
				time[moment] = time[moment]+'0';
			}
		} else if (time[moment] < 10) {
			time[moment] = '0' + time[moment];
		}
	});

	var str = y + '/' + mo + '/' + d + ' ' + time.h + ':' + time.mi + ':' + time.s + '.' + time.ms;
	if (TS_last_log_date) {
		var diff = date-TS_last_log_date;
		//str+= ' ('+diff+'ms)';
	}
	TS_last_log_date = date;
	return str+' ';
}

var parseDeepLinkRequest = function(code) {
	var m = code.match(/"id":"([CDG][A-Z0-9]{8})"/);
	var id = m ? m[1] : null;

	m = code.match(/"team":"(T[A-Z0-9]{8})"/);
	var team = m ? m[1] : null;

	m = code.match(/"message":"([0-9]+\.[0-9]+)"/);
	var message = m ? m[1] : null;

	return { id: id, team: team, message: message };
}

if ('rendererEvalAsync' in window) {
	var origRendererEvalAsync = window.rendererEvalAsync;
	window.rendererEvalAsync = function(blob) {
		try {
			var data = JSON.parse(decodeURIComponent(atob(blob)));
			if (data.code.match(/handleDeepLink/)) {
				var request = parseDeepLinkRequest(data.code);
				if (!request.id || !request.team || !request.message) return;

				request.cmd = 'channel';
				TSSSB.handleDeepLinkWithArgs(JSON.stringify(request));
				return;
			} else {
				origRendererEvalAsync(blob);
			}
		} catch (e) {
		}
	}
}
</script>



<script type="text/javascript">

	var TSSSB = {
		call: function() {
			return false;
		}
	};

</script>
<script>TSSSB.env = (function() {


	var v = {
		win_ssb_version: null,
		win_ssb_version_minor: null,
		mac_ssb_version: null,
		mac_ssb_version_minor: null,
		mac_ssb_build: null,
		lin_ssb_version: null,
		lin_ssb_version_minor: null,
		desktop_app_version: null,
	};

	var is_win = (navigator.appVersion.indexOf('Windows') !== -1);
	var is_lin = (navigator.appVersion.indexOf('Linux') !== -1);
	var is_mac = !!(navigator.userAgent.match(/(OS X)/g));

	if (navigator.userAgent.match(/(Slack_SSB)/g) || navigator.userAgent.match(/(Slack_WINSSB)/g)) {
		
		var parts = navigator.userAgent.split('/');
		var version_str = parts[parts.length - 1];
		var version_float = parseFloat(version_str);
		var version_parts = version_str.split('.');
		var version_minor = (version_parts.length == 3) ? parseInt(version_parts[2], 10) : 0;

		if (navigator.userAgent.match(/(AtomShell)/g)) {
			
			if (is_lin) {
				v.lin_ssb_version = version_float;
				v.lin_ssb_version_minor = version_minor;
			} else if (is_win) {
				v.win_ssb_version = version_float;
				v.win_ssb_version_minor = version_minor;
			} else if (is_mac) {
				v.mac_ssb_version = version_float;
				v.mac_ssb_version_minor = version_minor;
			}

			if (version_parts.length >= 3) {
				v.desktop_app_version = {
					major: parseInt(version_parts[0], 10),
					minor: parseInt(version_parts[1], 10),
					patch: parseInt(version_parts[2], 10),
				};
			}
		}
	}

	return v;
})();
</script>


	<script type="text/javascript">
		
		window.addEventListener('load', function() {
			var was_TS = window.TS;
			delete window.TS;
			if (was_TS) window.TS = was_TS;
		});
	</script>
	        <title>lab11finished.py | BYU Deep Learning Fall 2017 Slack</title>
    <meta name="author" content="Slack">
        

	
		
	
	
		
	
						
	
	

	
	
	
	
	
	
	
		<!-- output_css "sk_adapter" -->
    <link href="https://a.slack-edge.com/ba82c/style/rollup-slack_kit_legacy_adapters.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

			<!-- output_css "core" -->
    <link href="https://a.slack-edge.com/71e43/style/rollup-plastic.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

		<!-- output_css "before_file_pages" -->
    <link href="https://a.slack-edge.com/74a30/style/libs/codemirror.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">
    <link href="https://a.slack-edge.com/2e1ec/style/codemirror_overrides.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

	<!-- output_css "file_pages" -->
    <link href="https://a.slack-edge.com/65557/style/rollup-file_pages.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

	
			<!-- output_css "slack_kit_helpers" -->
    <link href="https://a.slack-edge.com/749a/style/rollup-slack_kit_helpers.css" rel="stylesheet" type="text/css" id="slack_kit_helpers_stylesheet"  crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

	<!-- output_css "regular" -->
    <link href="https://a.slack-edge.com/25e57/style/print.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">
    <link href="https://a.slack-edge.com/181a56/style/libs/lato-2-compressed.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

	

	
	
		<meta name="robots" content="noindex, nofollow" />
	

	
<link id="favicon" rel="shortcut icon" href="https://a.slack-edge.com/436da/marketing/img/meta/favicon-32.png" sizes="16x16 32x32 48x48" type="image/png" />

<link rel="icon" href="https://a.slack-edge.com/436da/marketing/img/meta/app-256.png" sizes="256x256" type="image/png" />

<link rel="apple-touch-icon-precomposed" sizes="152x152" href="https://a.slack-edge.com/436da/marketing/img/meta/ios-152.png" />
<link rel="apple-touch-icon-precomposed" sizes="144x144" href="https://a.slack-edge.com/436da/marketing/img/meta/ios-144.png" />
<link rel="apple-touch-icon-precomposed" sizes="120x120" href="https://a.slack-edge.com/436da/marketing/img/meta/ios-120.png" />
<link rel="apple-touch-icon-precomposed" sizes="114x114" href="https://a.slack-edge.com/436da/marketing/img/meta/ios-114.png" />
<link rel="apple-touch-icon-precomposed" sizes="72x72" href="https://a.slack-edge.com/436da/marketing/img/meta/ios-72.png" />
<link rel="apple-touch-icon-precomposed" href="https://a.slack-edge.com/436da/marketing/img/meta/ios-57.png" />

<meta name="msapplication-TileColor" content="#FFFFFF" />
<meta name="msapplication-TileImage" content="https://a.slack-edge.com/436da/marketing/img/meta/app-144.png" />
	
</head>

<body class="				">

		  			<script>
		
			var w = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
			if (w > 1440) document.querySelector('body').classList.add('widescreen');
		
		</script>
	
  	
	

			

<nav id="site_nav" class="no_transition">

	<div id="site_nav_contents">

		<div id="user_menu">
			<div id="user_menu_contents">
				<div id="user_menu_avatar">
										<span class="member_image thumb_48" style="background-image: url('https://secure.gravatar.com/avatar/5b41c07f05794548c219bab9a030c2e0.jpg?s=192&d=https%3A%2F%2Fa.slack-edge.com%2F7fa9%2Fimg%2Favatars%2Fava_0005-192.png')" data-thumb-size="48" data-member-id="U71EJ5FT9"></span>
					<span class="member_image thumb_36" style="background-image: url('https://secure.gravatar.com/avatar/5b41c07f05794548c219bab9a030c2e0.jpg?s=72&d=https%3A%2F%2Fa.slack-edge.com%2F66f9%2Fimg%2Favatars%2Fava_0005-72.png')" data-thumb-size="36" data-member-id="U71EJ5FT9"></span>
				</div>
				<h3>Signed in as</h3>
				<span id="user_menu_name">jaringson</span>
			</div>
		</div>

		<div class="nav_contents">

			<ul class="primary_nav">
									<li><a href="/messages" data-qa="app"><i class="ts_icon ts_icon_angle_arrow_up_left"></i>Back to Slack</a></li>
								<li><a href="/home" data-qa="home"><i class="ts_icon ts_icon_home"></i>Home</a></li>
				<li><a href="/account" data-qa="account_profile"><i class="ts_icon ts_icon_user"></i>Account &amp; Profile</a></li>
				<li><a href="/apps/manage" data-qa="configure_apps" target="_blank"><i class="ts_icon ts_icon_plug"></i>Configure Apps</a></li>
									<li><a href="/files" data-qa="files"><i class="ts_icon ts_icon_all_files clear_blue"></i>Files</a></li>
				
														<li><a href="/stats" data-qa="statistics"><i class="ts_icon ts_icon_dashboard"></i>Analytics</a></li>
													<li><a href="/customize" data-qa="customize"><i class="ts_icon ts_icon_magic"></i>Customize</a></li>
													<li><a href="/account/team" data-qa="team_settings"><i class="ts_icon ts_icon_cog_o"></i>Workspace Settings</a></li>
							</ul>

			
		</div>

		<div id="footer">

			<ul id="footer_nav">
				<li><a href="/is" data-qa="tour">Tour</a></li>
				<li><a href="/downloads" data-qa="download_apps">Download Apps</a></li>
				<li><a href="/brand-guidelines" data-qa="brand_guidelines">Brand Guidelines</a></li>
				<li><a href="/help" data-qa="help">Help</a></li>
				<li><a href="https://api.slack.com" target="_blank" data-qa="api">API<i class="ts_icon ts_icon_external_link small_left_margin ts_icon_inherit"></i></a></li>
								<li><a href="/pricing?ui_step=96&ui_element=5" data-qa="pricing" data-clog-event="GROWTH_PRICING" data-clog-ui-element="pricing_link" data-clog-ui-action="click" data-clog-ui-step="admin_footer">Pricing</a></li>
				<li><a href="/help/requests/new" data-qa="contact">Contact</a></li>
				<li><a href="/terms-of-service" data-qa="policies">Policies</a></li>
				<li><a href="https://slackhq.com/" target="_blank" data-qa="our_blog">Our Blog</a></li>
				<li><a href="https://slack.com/signout/238068740516?crumb=s-1512071074-95f8cf868b-%E2%98%83" data-qa="sign_out">Sign Out<i class="ts_icon ts_icon_sign_out small_left_margin ts_icon_inherit"></i></a></li>
			</ul>

			<p id="footer_signature">Made with <i class="ts_icon ts_icon_heart"></i> by Slack</p>

		</div>

	</div>
</nav>	
			
<header>
			<a id="menu_toggle" class="no_transition" data-qa="menu_toggle_hamburger">
			<span class="menu_icon"></span>
			<span class="menu_label">Menu</span>
			<span class="vert_divider"></span>
		</a>
		<h1 id="header_team_name" class="inline_block no_transition" data-qa="header_team_name">
			<a href="/home">
				<i class="ts_icon ts_icon_home" /></i>
				BYU Deep Learning Fall 2017
			</a>
		</h1>
		<div class="header_nav">
			<div class="header_btns float_right">
				<a id="team_switcher" data-qa="team_switcher">
					<i class="ts_icon ts_icon_th_large ts_icon_inherit"></i>
					<span class="block label">Workspaces</span>
				</a>
				<a href="/help" id="help_link" data-qa="help_link">
					<i class="ts_icon ts_icon_life_ring ts_icon_inherit"></i>
					<span class="block label">Help</span>
				</a>
															<a href="/messages" data-qa="launch">
							<img src="https://a.slack-edge.com/66f9/img/icons/ios-64.png" srcset="https://a.slack-edge.com/66f9/img/icons/ios-32.png 1x, https://a.slack-edge.com/66f9/img/icons/ios-64.png 2x" title="Slack" alt="Launch Slack"/>
							<span class="block label">Launch</span>
						</a>
												</div>
							<ul id="header_team_nav" data-qa="team_switcher_menu">
																										
<li class="active">
	<a href="https://byu-dl-f17.slack.com/home" target="https://byu-dl-f17.slack.com/">
					<i class="ts_icon small ts_icon_check_circle_o active_icon s"></i>
							<i class="team_icon small default" >BD</i>
				<span class="switcher_label team_name">BYU Deep Learning Fall 2017</span>
	</a>
</li>															
<li >
	<a href="https://mars-rover.slack.com/home" target="https://mars-rover.slack.com/">
							<i class="team_icon small" style="background-image: url('https://slack-files2.s3-us-west-2.amazonaws.com/avatars/2017-04-15/169734687875_9e15f4d191d5fa48593e_88.png');"></i>
				<span class="switcher_label team_name">Mars Rover</span>
	</a>
</li>															
<li >
	<a href="https://byumarsrover2018.slack.com/home" target="https://byumarsrover2018.slack.com/">
							<i class="team_icon small" style="background-image: url('https://slack-files2.s3-us-west-2.amazonaws.com/avatars/2017-09-22/245086437553_0037315a2e071ebd9528_88.png');"></i>
				<span class="switcher_label team_name">BYU Mars Rover 2018</span>
	</a>
</li>																							<li id="add_team_option"><a href="https://slack.com/signin" target="_blank"><i class="ts_icon ts_icon_plus team_icon small"></i> <span class="switcher_label">Sign in to another workspace &hellip;</span></a></li>
				</ul>
					</div>
	
	
	
</header>	
	<div id="page" >

		<div id="page_contents" data-qa="page_contents" class="">


<p class="print_only">
	<strong>
	
	Created by joseph on Wednesday, November 29, 2017 at 8:46 PM
	</strong><br />
	<span class="subtle_silver break_word">https://byu-dl-f17.slack.com/files/U71477WFQ/F88JBS1KR/lab11finished.py</span>
</p>

<div class="file_header_container no_print"></div>

<div class="alert_container">
		

<div class="file_public_link_shared alert" style="display: none;">
		
	<i class="ts_icon ts_icon_link"></i> Public Link: <a class="file_public_link" href="https://slack-files.com/T7020MSF6-F88JBS1KR-2578eac0a9" target="new">https://slack-files.com/T7020MSF6-F88JBS1KR-2578eac0a9</a>
</div></div>

<div id="file_page" class="card top_padding">

	<p class="small subtle_silver no_print meta">
		
		23KB Python snippet created on <span class="date">November 29, 2017</span>.
						<span class="file_share_list"></span>
	</p>

	<a id="file_action_cog" class="action_cog action_cog_snippet float_right no_print">
		<span>Actions </span><i class="ts_icon ts_icon_cog"></i>
	</a>
	<a id="snippet_expand_toggle" class="float_right no_print">
		<i class="ts_icon ts_icon_expand "></i>
		<i class="ts_icon ts_icon_compress hidden"></i>
	</a>

	<div class="large_bottom_margin clearfix">
		<pre id="file_contents">#!/usr/bin/python
# -*- coding: utf-8

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from itertools import chain
from numpy.random import randint
import re


def train_test_split(source_file, target_file, n_test=3000, max_sentence_len=30):
    &quot;&quot;&quot;
    Both text files contain sentences in each language and map between
      each other one to one.
    This method reads the files into memory after eliminating all
      nonstandard english characters.
    Then it ensures that no sentence pairs are kept where one or the other exceeeds
      the maximum sentence length.
    A random sample of the data is set aside for training/validation.
    Corpus objects for both languages are created
    -obtains a list of unique words
    -assigns each an index
    -these along with EOS and SOS are the tokens that
     will be used for training.
    Like the Char-RNN lab using the indices generated in each corpus 
     each sentence&#039;s words are mapped to indexes.
    returns:
      testing_pairs: A list of the original sentence pairs (a list of tuples 
                       of two strings each [(spanish_sentence1, english_sentence1), ...] 
      training_pairs: similar but for training
      train_index_pairs: a list of tuples containing the indices of the words
                         in each sentence in the training_pairs 
                         [ ([11, 1592, 350, 279, 438], [2, 219, 38]), ... ]
      test_index_pairs: similar but for training_pairs
      
    If you don&#039;t like working with a list of tuples just call
      spanish_testing, english_testing = zip(*testing_pairs) # both lists of strings now
      spanish_training, english_training = zip(*training_pairs) # same
      
      # the following are lists of unequal length integer lists representing indices 
      indexed_spanish_training, indexed_english_training = zip(*train_index_pairs) 
      indexed_spanish_testing, indexed_english_testing = zip(*test_index_pairs)
     
     The only difference until now from the Char-RNN lab is in processing data for two languages
       and limiting the data to sentences that aren&#039;t above a certain length.
    &quot;&quot;&quot;
    
    # reads file and removes all non a-z characters
    def read_file(text_file):
        r1 = lambda x:re.sub(r&quot;[.!?]$&quot;, r&quot; &lt;EOS&gt;&quot;, x.strip().lower())
        r2 = lambda x:re.sub(r&quot;(-|—)&quot;, &quot; &quot;, x)
        r3 = lambda x:re.sub(r&quot;\([^\(]+\)&quot;, &quot; &quot;, x)
        
        contents = [r1(r2(r3(l))) for l in open(text_file, &quot;r&quot;)]
        acceptable = list(range(97,123)) + \
                     [32, 160, 193, 201, 205, 209, 211, 218, 220,\
                      225, 233, 237, 241, 243, 250, 252]

        contents = [filter(lambda x:ord(x) in acceptable, \
                           line).strip().lower() for line in contents]
        contents = [&quot; &quot;.join(re.split(r&quot;\s+&quot;, line)) for line in contents] # removes consecutive spaces
        return contents

    source_lines = read_file(source_file)
    target_lines = read_file(target_file)
    
    n_words = lambda s:len(s.split(&quot; &quot;))
   
        
   
        
    keep_pair_if = lambda pair: 6 &lt; min(n_words(pair[0]),n_words(pair[1])) &lt; max_seq_len and \
                                len(re.findall(&quot;(http|www|org|edu|com)&quot;, pair[1])) == 0
        
    pairs = zip(source_lines, target_lines)
    pairs = filter(keep_pair_if, pairs)

    # could try training on shortest first. 
    #shortest = lambda pair: min(n_words(pair[0]),n_words(pair[1]))
    #pairs = sorted(pairs, key=shortest)
    
    # filter keeps all elements of pairs
    pairs = filter(keep_pair_if, pairs)
    source_lines, target_lines = zip(*pairs)
    print len(source_lines), &#039;lines in data&#039;

    source_corpus = Corpus(source_lines)
    target_corpus = Corpus(target_lines)

    n_spanish = source_corpus.corpus_size
    n_english = target_corpus.corpus_size
    all_indexed_pairs = zip(source_corpus.training, target_corpus.training)

    np.random.seed(2)
    test_idc = randint(0, len(pairs), n_test)
    train_idc = set(np.arange(len(pairs))) - set(test_idc)

    # list of 2-string tuples consisting of source / reference sentence pairs
    testing_pairs = map(lambda k: pairs[k], test_idc)
    training_pairs = map(lambda k: pairs[k], train_idc)

    # list of tuples consisting of source / reference sentence pairs 
    # but represented by word indexes
    train_index_pairs = map(lambda k:all_indexed_pairs[k], train_idc)
    test_index_pairs = map(lambda k:all_indexed_pairs[k], test_idc)
    
    return source_lines, target_lines, source_corpus, target_corpus, testing_pairs, training_pairs, train_index_pairs, test_index_pairs


class Corpus():
    def __init__(self, input_lines, n_train=5000):
        self.SOS = 0
        self.EOS = 1
        self.idx_word,         self.word_idx = self.parse_words(input_lines)
        self.n_train = n_train
        
        self.parse_words(input_lines)
        self.corpus_size = len(self.idx_word)
        self.lines = [l.strip().lower() for l in input_lines]
        self.training = [self.sentence_to_index(l) for l in self.lines]
        
    def parse_words(self, lines):
        sls = lambda s: s.strip().lower().split(&quot; &quot;)
        words = sorted(set(list(                 chain(*[sls(l) for l in lines]))))
        words = [&quot;&lt;SOS&gt;&quot;, &quot;&lt;EOS&gt;&quot;] + filter(lambda word:set(word).intersection(set(map(chr, range(97,123))))!=set(), words)
        n = 3000
        print(len(words), &#039;words&#039;)
        idx_word = dict(list(enumerate(words)))
        word_idx = dict(zip(words, list(range(len(words)))))
        
        return idx_word, word_idx
    
    def sentence_to_index(self, s):
        words = s.split(&quot; &quot;)
        indices = [self.word_idx[word] for word in words]
        return indices
    
    def index_to_sentence(self, indices):
        return &quot; &quot;.join(self.idx_word[idx] for idx in indices)


# recommend to use this class later, same as nn.Linear
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.W = torch.nn.Parameter(torch.randn(output_size, input_size))
        self.b = torch.nn.Parameter(torch.randn(output_size, 1))
        
        # use rand(n) to get tensors to intitialize your weight matrix and bias tensor 
        # then use Parameter( ) to wrap them as Variables visible to module.parameters()
        # to use variance scaling initialization just divide the variable like a ndarray
        
    def forward(self, input_var):
        # standard linear layer, no nonlinearity
        return torch.matmul(self.W, input_var) + self.b


&quot;&quot;&quot;
encoder topology
-standard GRU topology, see slides for a reveiw
-context vector is the hidden state of the last time step and last layer
-review the char-rnn lab

below every GRUCell is its input, to its left is the hidden input
-note that of all the outputs and final hidden states only one is kept

-h_3 -&gt; GRUCell3 -&gt; GruCell3 -&gt; ... -&gt; GRUCell3 -&gt; context vector
-h_2 -&gt; GRUCell2 -&gt; GruCell2 -&gt; ... -&gt; GRUCell2
-h_1 -&gt; GRUCell1 -&gt; GruCell1 -&gt; ... -&gt; GRUCell1
        emb[0]      emb[1]             emb[n-1]

-use zero Variables as the initial hidden states

Pytorch RNN pipelines and this lab
-Never use one hot encodings in pytorch. Several loss functions, nn.Embedding, 
 and other classes are programmed to use indexed tensors whenever possible
-like tensorflow and GRUCell  (batch, input_dim), (batch, hidden_dim) shaped tensor Variables
 in our case batch = 1 and input_dim = hidden_dim most likely
-nn.Embeddding does the same thing as one hot encoding the input and then running
  it through a linear transformation (so no bias or nonlinearity)
-good idea to specify the max_norm of both embeddings when initialized
&quot;&quot;&quot;


class Encoder(nn.Module):
    def __init__(self, hidden_size, source_vocab_size, n_layers=2):
        super(Encoder, self).__init__()
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = source_vocab_size
        self.n_layers = n_layers
        
        # multiple ways to do this
        cells = [nn.GRUCell(self.hidden_size, self.hidden_size) for _ in range(n_layers)]
        self.cells = nn.ModuleList(cells)        
        self.GRU = nn.GRU(self.input_size, self.hidden_size, self.n_layers)
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        
        self.embedding = nn.Embedding(self.vocab_size, self.input_size, scale_grad_by_freq=True)
        
        # so they&#039;re accessible to decoder.cuda()
        self.init_hidden_states = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        self.init_cell_states = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    

    # if you want to udnerstand this better consider writing
    # implementing tf.contrib.rnn.legacy.MultiCellRNN here
    def time_step(self, input_, hidden_states):
        pass
    
    # consider coding this as rnn_decoder as shown in the Char-RNN lab
    def forward(self, source_variable):
        # discard all outputs except the last one.
        n_timesteps = source_variable.data.cpu().size()[0]
        embedded = self.embedding(source_variable).view(-1, 1, self.hidden_size)
        hidden_states = self.init_hidden_states
        cell_states = self.init_cell_states
        
        &quot;&quot;&quot; 1st implementation watch shapes
        hidden_states = [torch.zeros(1, self.hidden_size) for _ in range(self.n_layers)]
        for i in range(n_timesteps):
            out = embedded[i] # GRUCell takes (batch, hidden_dim) not (batch, seq_len, hidden_size)
            for j in range(self.n_layers):
                hidden_states[j] = self.cells[j](out, hidden_states[j])
                out = hidden_states[j]
        return out&quot;&quot;&quot;


        &quot;&quot;&quot; 2nd implementation
        for i in range(n_timesteps):
             out, hidden_states = self.GRU(self.embedding[i:i+1], hidden_states) 
             #out, (hidden_states, cell_states) = self.LSTM(self.embedding[i:i+1], (hidden_states, cell_states)) 
        
        return out &quot;&quot;&quot;
        
        # 3rd
        return self.GRU(self.embedding(source_variable).view(-1, 1, self.hidden_size), hidden_states)[0][-1]
        #return self.LSTM(self.embedding(source_variable).view(-1, 1, self.hidden_size), \
        #                 (hidden_states, cell_states))[0][-1]
   

class Decoder(nn.Module):
    def __init__(self, hidden_size, target_vocab_size, n_layers=2, max_target_length=30):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.max_length = max_target_length
        
        # three possible implementations
        self.cells = nn.ModuleList([nn.GRUCell(self.input_size, self.hidden_size) for _ in range(n_layers)])
        self.GRU = nn.GRU(self.input_size, self.hidden_size, self.n_layers)
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        
        # maps to logits
        self.Linear = Linear(self.hidden_size, self.vocab_size)
        # much better to store words in 1000 dimensional space than in 20000 (one hot encoded) dim space
        self.embedding = nn.Embedding(self.vocab_size, hidden_size, scale_grad_by_freq=True)

        self.loss = nn.CrossEntropyLoss() # here so it&#039;s accessible using decoder.cuda()

        
    def forward(self, context, reference_sentence=None):
        # if the reference sentence is given then use it for teacher forcing
        use_teacher_forcing = reference_sentence is not None

        embedded_sos = self.embedding(Variable(torch.LongTensor([[0]]))).view(1, 1, self.hidden_size)

        predictions = []                           
        output_logits= []

        if use_teacher_forcing:
            inputs = [embedded_sos.view(1, 1, hidden_size)] +                      list(torch.split(self.embedding(reference_sentence)                                .view(-1, 1, self.hidden_size)[1:], 1, 0))
            input_ = inputs[0]
        else:
            input_ = embedded_sos

        c_i = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        h_i = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        
        for i in range(self.max_length):            
            #output, (h_i, c_i) = self.LSTM(input_, (h_i, c_i))
            output, h_i = self.GRU(input_, h_i)

            logits = self.Linear(output.view(-1, 1))
            prediction = logits.data.cpu().numpy().argmax()

            predictions.append(prediction)
            if not use_teacher_forcing:
                if prediction == target_corpus.EOS:
                    break
                
                input_ = self.embedding(Variable(torch.LongTensor([[prediction]])))
            else:
                if i == reference_sentence.size()[0]-1:
                    break # don&#039;t translate for longer
                input_ = inputs[i+1]
                
            output_logits.append(logits)

        return output_logits, predictions


# changed
def get_loss(output_probs, correct_sentence_indices, predicted_sentence_indices, loss_op):
    &quot;&quot;&quot; 
    note that output_probs and correct_indices will often have different shapes
    look up documentaiton on NLLLoss, but if you 
    predicted_inde
    inputs:
    output_probs Variable
    target_sentence_indices
    predicted_sentence_indices
    params:
      output_probs: a list of Variable (not FloatTensor) with the predicted sequence length
      correct_indices: a list or tensor of type int with the same length, will need
                       to be converted to a Variable before compared to the output_probs
    
    &quot;&quot;&quot;
    
    # changed: Convert correct_sentence_indices to a Variable if it isn&#039;t already
    # remember embedded tensors (like your logit probabilities) should have one more dimension than
    #    the indexed tensor that they&#039;re compared against
    # compute cross entropy, recommend NLLLoss since it works after LogSoftmax
    #  approx need to call log  on your logit softmax probabilities or use 
    #  max score or LogSoftmax closest to zero for predictions
    # should return a Variable representing the loss
    # reommended: consider light dropout before L1, L2 regularization.
    
    # moved loss to decoder so it would be accessible to decoder.cuda()
 
    if type(output_probs) is list:
        output_probs = torch.cat(output_probs, dim=1).transpose(0,1)
        
    min_length = int(min(output_probs.size()[0], correct_sentence_indices.size()[0]))
    loss = loss_op(output_probs[0:min_length], correct_sentence_indices[0:min_length])
    
    arr1 = correct_sentence_indices.data.cpu().numpy()
    arr2 = np.array(predicted_sentence_indices) # should have been list of ints
    accuracy = float(np.sum(arr1[0:min_length] == arr2[0:min_length])) / arr1.shape[0]

    return loss, accuracy 


def print_output(epoch=None, in_out_ref_sentences=None, stats=None, teacher_forced=False):
    &quot;&quot;&quot;
    params:
      teacher_forced: whether or not teacher forcing was used
    &quot;&quot;&quot;  
    # not a great practice to use global
    global source_corpus, target_corpus
    
    if stats is not None:
        s = &quot;: perplexity: %.3f: loss: %.6f, accuracy: %.1f&quot; % (2**stats[0], stats[0], stats[1])
        if teacher_forced:
            print(&quot;epoch %d %s - using teacher forcing&quot; % (epoch, s))
        else:
            print(&quot;epoch %d iteration %s&quot; % (epoch, s))
    
    if in_out_ref_sentences is not None:
        if epoch is not None:
            print (&quot;Outputs during epoch %d %s&quot; % (epoch, &quot;&quot; if not teacher_forced else &quot;using teacher forcing&quot;))

        source_indices, predicted_indices, reference_indices = in_out_ref_sentences
        print &quot;In:       &quot;, source_corpus.index_to_sentence(source_indices)
        print &quot;Out:      &quot;, target_corpus.index_to_sentence(predicted_indices)
        print &quot;Reference:&quot;, target_corpus.index_to_sentence(reference_indices)
        
        
def train(encoder, decoder, training_pairs, testing_pairs, train_index_pairs, test_index_pairs,
                source_corpus, target_corpus, teacher_forcing_ratio, 
                epoch_size, learning_rate, decay, batch_size, print_every):
    &quot;&quot;&quot;
    Again teacher forcing not required, but will reduce training time and is 
      much simpler to implement.
    You may want to lower the teacher forcing ratio as the number 
      of epochs progresses as it starts to learn word-word connections.
    
    If you wish to use a learning rate schedule, you will need to initialize new optimizers
       every epoch. Note that only a few optimizers let you specify learning rate decay.
    See notes below for help in training / debugging
    Don&#039;t hesitate to ask for help1
    
    &quot;&quot;&quot;
    if torch.cuda.is_available():
        arr2var = lambda sent: Variable(torch.LongTensor(sent)).cuda()
    else:
        arr2var = lambda sent: Variable(torch.LongTensor(sent))
 
    n_test = len(testing_pairs)
    training_var_pairs = [(arr2var(_1), arr2var(_2)) for (_1, _2) in train_index_pairs]
    testing_var_pairs =  [(arr2var(_1), arr2var(_2)) for (_1, _2) in test_index_pairs]
    
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(all_params, lr=learning_rate, weight_decay=decay)     
    
    # if training on increasing length sentences
    # sentence_id = 0
    batch_loss, batch_acc = [], [] # for printing
    
    for i in range(n_epochs):
        for j in range(epoch_size):
            # consider whether or not to use teacher forcing on printing iterations
            use_teacher_forcing = np.random.random() &lt; teacher_forcing_ratio
            
            sentence_id = np.random.randint(0, len(training_pairs))
                
            source, reference = training_var_pairs[sentence_id]
            context = encoder(source)
            ref = reference if use_teacher_forcing else None
            
            # list of Variables, list of int
            output_logits, predictions = decoder(context, reference_sentence=ref)
            
            loss, accuracy = get_loss(output_logits, reference, predictions, decoder.loss)
            
            loss.backward()

            # for reporting teacher forcing doesn&#039;t count towards batch training statistics
            if not use_teacher_forcing:
                batch_loss.append(loss.data.cpu().numpy().flatten()[0])
                batch_acc.append(accuracy)
            
            # we could have done loss /= batch_size; loss.backward()
            if (j+1) % batch_size == 0:
                # effective batch updates
                for p in all_params:
                    p.data.div_(batch_size)
                    
                optim.step()
                optim.zero_grad()

                batch_loss_, batch_loss = sum(batch_loss) / (len(batch_loss)+1e-13), []
                batch_acc_, batch_acc = sum(batch_acc) / (len(batch_acc)+1e-13), []

                print(&quot;\nEnd of batch %d epoch %d&quot; % (j // batch_size, i+1))
                print_output(epoch=i, stats=(batch_loss_, batch_acc_))
            
            if (j+1) % print_every == 0:
                source_idc, reference_idc = train_index_pairs[sentence_id]
                print_output(in_out_ref_sentences = (source_idc, predictions, reference_idc), teacher_forced=use_teacher_forcing)
        
            if (j+1) % test_every == 0:
                test_loss, test_acc = 0, 0
                for ((src, tgt), (src_str, tgt_str)) in zip(testing_var_pairs, test_index_pairs):
                    probs, predictions = decoder(encoder(src))
                    _1, _2 = get_loss(probs, tgt, predictions, decoder.loss)
                    test_loss += _1.data.cpu().numpy().flatten()[0]
                    test_acc += _2
                test_loss /= n_test
                test_acc /= n_test

                print(&quot;\n-- Epoch %d Test Results;: perplexity: %.3f: loss: %.6f, accuracy: %.1f\n&quot; % (i+1, 2**test_loss, test_loss, test_acc))

    return encoder, decoder

# can use this to print out your final translation sentences 
def sample(encoder, decoder, sentence_pairs):#, testing_results = None):
    # never use volatile outside of inference
    if torch.cuda.is_available():
        arr2var = lambda x: Variable(torch.LongTensor(x), volatile=True).cuda()
    else:
        arr2var = lambda x: Variable(torch.LongTensor(x), volatile=True)
    
    sentence_var_pairs =  [(arr2var(_1), arr2var(_2)) for (_1, _2) in sentence_pairs]
    prin
    for (source_sentence, reference_sentence) in sentence_var_pairs:
        context = encoder(source_sentence)
        output_probs, predictions = decoder(reference_sentence)
        print_output(in_out_ref_sentences=(source_sentence, predictions, reference_sentence))

# parameters needed for data processing
source_file = &quot;data/es.txt&quot;
target_file = &quot;data/en.txt&quot;
n_test = 30
max_seq_len = 30 # maximum sentence length

# pretty much everything you need, see train_test_split doc string
source_lines, target_lines, source_corpus, target_corpus, testing_pairs, training_pairs, train_index_pairs, test_index_pairs = train_test_split(source_file, target_file, n_test, max_seq_len)

# example hyperparameters
epoch_length = 800
batch_size = 20
n_layers = 2  # 2 or more recommended
learning_rate = .01
decay_rate = .98 # decays every batch_size sentences
print_every = 3
n_epochs = 25
hidden_size = 1024
teacher_forcing_ratio = .5

encoder = Encoder(hidden_size, source_corpus.corpus_size, n_layers)
decoder = Decoder(hidden_size, target_corpus.corpus_size, n_layers, max_target_length=max_seq_len)
use_cuda = torch.cuda.is_available()

# all of our submodules and Variables are class members of encoder and decoder
# so this is the only time we need to call this. If you are using a gpu ensure
# you call .cpu() whenever you want to directly access a tensor&#039;s value from code
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

encoder, decoder = train(encoder, decoder, training_pairs, testing_pairs, \
                         train_index_pairs, test_index_pairs, source_corpus, \
                         target_corpus, teacher_forcing_ratio, epoch_length, \
                         learning_rate, decay_rate, batch_size, print_every)


_=&quot;&quot;&quot;
Start simple, seriously
-consider implementing it without teacher forcing first.
-just get a scalar loss variable as fast as you can.

Recommend writing new code first inside of a jupyter notebook cell
-Just initialize variables of the correct size as mock inputs
-Can see the Variables after each line of code
-this way getting Variable shapes right will be less of a challenge
-Can use tab for code completion and shift tab to see the signature of
   any torch method when inside the parentheses
-About a majority of the methods are the same as numpy or 
 tensorflow so chances are with a few tries you&#039;ll find the right method
-This is also a very good way to learn pytorch

If and only if it&#039;s not working
-start by tuning hyperparameters
-batch_size, teacher forcing, learning rate are good places to start
-Make sure your teacher forcing implementation is correct

Training philosophy behind vanilla seq2seq nmt systems (see also sutskever, 2014):
-need to learn somewhat one to one word connections first
-shift to learning long term dependencies by reducing teacher forcing 
-consider a schedule for learning rate and/or teacher forcing ratio

Saving and restoring weights might speed up your workflow significantly
-might want to checkpoint once you&#039;ve learned word-word connections

Regularization is super simple in pytorch.
-Be careful woing dropout on the hidden states between cells
  see https://arxgiv.org/pdf/1512.05287.pdf.
-low to moderate dropout after both embeddings may be helpful
decoder = Decoder(hidden_size, target_corpus.corpus_size, n_layers, max_target_length=max_seq_len)&quot;&quot;&quot;
</pre>

		<p class="file_page_meta no_print" style="line-height: 1.5rem;">
			<label class="checkbox normal mini float_right no_top_padding no_min_width">
				<input type="checkbox" id="file_preview_wrap_cb"> wrap long lines			</label>
		</p>

	</div>

			<div id="comments_holder" class="clearfix clear_both">
	<div class="col span_1_of_6"></div>
	<div class="col span_4_of_6 no_right_padding">
		<div id="file_page_comments">
			

<div class="loading_hash_animation">
	<span class=loading_hash_container><img src="https://a.slack-edge.com/9c217/img/loading_hash_animation_@2x.gif" srcset="https://a.slack-edge.com/9c217/img/loading_hash_animation.gif 1x, https://a.slack-edge.com/9c217/img/loading_hash_animation_@2x.gif 2x" alt="Loading" class="loading_hash" /><br />loading...</span>
	<noscript>
		
			You must enable javascript in order to use Slack :(
						<style type="text/css">span.loading_hash_container { display: none; }</style>
	</noscript>
</div>		</div>	
		

	<p class="p-external_file_author_notice hidden">
		
			<strong class="dull_grey">Can’t view comments</strong><br />
			This file is from another workspace.
			</p>

	<form action="https://byu-dl-f17.slack.com/files/U71477WFQ/F88JBS1KR/lab11finished.py"
			id="file_comment_form"
							class="comment_form clearfix"
						method="post">
					<a href="/team/U71EJ5FT9" class="member_preview_link" data-member-id="U71EJ5FT9" >
				<span class="member_image thumb_36" style="background-image: url('https://secure.gravatar.com/avatar/5b41c07f05794548c219bab9a030c2e0.jpg?s=72&d=https%3A%2F%2Fa.slack-edge.com%2F66f9%2Fimg%2Favatars%2Fava_0005-72.png')" data-thumb-size="36" data-member-id="U71EJ5FT9"></span>
			</a>
				<input type="hidden" name="addcomment" value="1" />
		<input type="hidden" name="crumb" value="s-1512071074-fd0bc8e0bf-☃" />

		<div id="file_comment" class="small texty_comment_input comment_input small_bottom_margin" name="comment" wrap="virtual" ></div>
		<div class="file_comment_submit_container display_flex justify_content_between">
			<button type="submit" class="file_comment_submit_btn btn align_self_start   ladda-button" data-style="expand-right"><span class="ladda-label">Add Comment</span></button>
			<span class="input_note indifferent_grey file_comment_tip">shift+enter to add a new line</span>		</div>
	</form>

<form
		id="file_edit_comment_form"
					class="edit_comment_form clearfix hidden"
				method="post">
		<div id="file_edit_comment" class="small texty_comment_input comment_input small_bottom_margin" name="comment" wrap="virtual"></div>
	<input type="submit" class="save btn float_right " value="Save" />
	<button class="cancel btn btn_outline float_right small_right_margin ">Cancel</button>
</form>	
	</div>
	<div class="col span_1_of_6"></div>
</div>	
</div>


	
		
	</div>
	<div id="overlay"></div>
</div>







<script type="text/javascript">

	/**
	 * A placeholder function that the build script uses to
	 * replace file paths with their CDN versions.
	 *
	 * @param {String} file_path - File path
	 * @returns {String}
	 */
	function vvv(file_path) {

		var vvv_warning = 'You cannot use vvv on dynamic values. Please make sure you only pass in static file paths.';
		if (TS && TS.warn) {
			TS.warn(vvv_warning);
		} else {
			console.warn(vvv_warning);
		}

		return file_path;
	}

	var cdn_url = "https:\/\/slack.global.ssl.fastly.net";
	var vvv_abs_url = "https:\/\/slack.com\/";
	var inc_js_setup_data = {
			emoji_sheets: {
			apple: 'https://a.slack-edge.com/bfaba/img/emoji_2016_06_08/sheet_apple_64_indexed_256colors.png',
			google: 'https://a.slack-edge.com/f360/img/emoji_2016_06_08/sheet_google_64_indexed_128colors.png',
			twitter: 'https://a.slack-edge.com/bfaba/img/emoji_2016_06_08/sheet_twitter_64_indexed_128colors.png',
			emojione: 'https://a.slack-edge.com/bfaba/img/emoji_2016_06_08/sheet_emojione_64_indexed_128colors.png',
		},
		};
</script>
	<script type="text/javascript">
<!--
	// common boot_data
	var boot_data = {
		start_ms: Date.now(),
		app: 'web',
		user_id: 'U71EJ5FT9',
		team_id: 'T7020MSF6',
		visitor_uid: ".5hvb3tig0cwsccc448skw84sc",
		no_login: false,
		version_ts: '1512069533',
		version_uid: '3b12e0970c59c74eaf054e23d960f9c4a0dcd138',
		cache_version: "v16-giraffe",
		cache_ts_version: "v2-bunny",
		redir_domain: 'slack-redir.net',
		signin_url: 'https://slack.com/signin',
		abs_root_url: 'https://slack.com/',
		api_url: '/api/',
		team_url: 'https://byu-dl-f17.slack.com/',
		image_proxy_url: 'https://slack-imgs.com/',
		beacon_timing_url: "https:\/\/slack.com\/beacon\/timing",
		beacon_error_url: "https:\/\/slack.com\/beacon\/error",
		clog_url: "clog\/track\/",
		api_token: 'xoxs-238068740516-239494185927-257070208578-c66b928e42',
		ls_disabled: false,

		vvv_paths: {
			
		lz_string: "https:\/\/a.slack-edge.com\/bv1-1\/lz-string-1.4.4.worker.8de1b00d670ff3dc706a0.js",
		codemirror: "https:\/\/a.slack-edge.com\/bv1-1\/codemirror.min.41c3faeb73621d67a666.min.js",
	codemirror_addon_simple: "https:\/\/a.slack-edge.com\/bv1-1\/simple.45192890ef119b00f332.min.js",
	codemirror_load: "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_load.0e1999f4ce5168fec6cb.min.js",

	// Silly long list of every possible file that Codemirror might load
	codemirror_files: {
		'apl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_apl.28ce658730a18a115532.min.js",
		'asciiarmor': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_asciiarmor.b6cae5185b1e92ac1917.min.js",
		'asn.1': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_asn.1.1992736a46ff4b01f93f.min.js",
		'asterisk': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_asterisk.8dc14a25826407ab1fa3.min.js",
		'brainfuck': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_brainfuck.d29773c7ac178228d4c1.min.js",
		'clike': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_clike.cccd21c94a2b7ab7ce3d.min.js",
		'clojure': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_clojure.4a91a0c50a64467f2ff5.min.js",
		'cmake': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_cmake.a873ff1604fb8e89955f.min.js",
		'cobol': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_cobol.2b7098fad4936f18f361.min.js",
		'coffeescript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_coffeescript.68a8fdd315d0dcf8c27a.min.js",
		'commonlisp': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_commonlisp.879f5b578b25a058fc4d.min.js",
		'css': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_css.035ca224464953138c30.min.js",
		'crystal': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_crystal.79beb330be1a294e43c4.min.js",
		'cypher': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_cypher.525ea24cf7710ac2825a.min.js",
		'd': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_d.7328ff9ab8c98103deb7.min.js",
		'dart': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dart.f7e22fcf397d31e7af93.min.js",
		'diff': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_diff.c3b6cf00600144071d6d.min.js",
		'django': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_django.cde302c62fe6365529f2.min.js",
		'dockerfile': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dockerfile.ed0e37e03b2023e1b69b.min.js",
		'dtd': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dtd.df3795754645134d5f80.min.js",
		'dylan': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dylan.fed72f1d0e846fd6d213.min.js",
		'ebnf': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ebnf.6af113fdedf587f96c64.min.js",
		'ecl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ecl.12b9206b24a5433649ab.min.js",
		'eiffel': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_eiffel.896ceb473406cc01ee59.min.js",
		'elm': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_elm.637ce7bda68e33c4b55a.min.js",
		'erlang': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_erlang.ea42edc0caacbb7b9429.min.js",
		'factor': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_factor.937f3b4ba675a2abe9c7.min.js",
		'forth': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_forth.89f6ec54d5548d4cf25b.min.js",
		'fortran': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_fortran.e312d7972b690a22beab.min.js",
		'gas': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_gas.abc6e9d7c3a0b887ff69.min.js",
		'gfm': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_gfm.8fc0c8e3735d10a858c6.min.js",
		'gherkin': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_gherkin.9e0cfa25c1965e495419.min.js",
		'go': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_go.5ed96d85ab19d7795ba7.min.js",
		'groovy': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_groovy.c496c31bd5cca0986ead.min.js",
		'haml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_haml.f25c65cf09f1ec3a29c7.min.js",
		'handlebars': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_handlebars.80eb7b9e2e0fb6dee382.min.js",
		'haskell': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_haskell.e7b2cc288c6bd281ff32.min.js",
		'haxe': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_haxe.3efebdfa3dc7fb7cc4db.min.js",
		'htmlembedded': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_htmlembedded.1a2496c621f9a470c340.min.js",
		'htmlmixed': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_htmlmixed.caa675603dc4fdb90c31.min.js",
		'http': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_http.c1c249d14bf18521fee1.min.js",
		'idl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_idl.715601d44fbe133c065b.min.js",
		'jade': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_jade.0eff9474d977d43feced.min.js",
		'javascript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_javascript.bc1b5c6a6515b3064108.min.js",
		'jinja2': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_jinja2.5de8bc52face9b2769f2.min.js",
		'julia': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_julia.32d8748fecc17e6305bf.min.js",
		'livescript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_livescript.f6dbad1e8d8168b319f4.min.js",
		'lua': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_lua.32780d85e5cbbb59b8eb.min.js",
		'markdown': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_markdown.a7f65f93ece1f31b9e60.min.js",
		'mathematica': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mathematica.48dd3694f2f71a75544c.min.js",
		'mirc': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mirc.0f3984162d72c3a0a5ca.min.js",
		'mllike': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mllike.e4e86126535bd4f7a575.min.js",
		'modelica': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_modelica.d4fd8938619f5c8dc361.min.js",
		'mscgen': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mscgen.f9d5ab8e95b697c39880.min.js",
		'mumps': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mumps.b361377133b59678d3d5.min.js",
		'nginx': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_nginx.524bfc39589c37f74bfd.min.js",
		'nsis': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_nsis.b25852c3418f984ae03d.min.js",
		'ntriples': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ntriples.4e0443a64025c35438a6.min.js",
		'octave': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_octave.3a0c99a5e07bbd9a6d67.min.js",
		'oz': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_oz.e9939d375a47087f59aa.min.js",
		'pascal': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_pascal.f1162aeab4a781363ccd.min.js",
		'pegjs': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_pegjs.7af50308b76ba3251b02.min.js",
		'perl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_perl.5a7940afe30ba510820a.min.js",
		'php': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_php.64b619fb529d1cd9b781.min.js",
		'pig': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_pig.a30ec6f3ffff33476ac4.min.js",
		'powershell': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_powershell.0ccd1b6a33eb2209c15b.min.js",
		'properties': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_properties.5c0c1436978bf2a7af00.min.js",
		'puppet': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_puppet.53ac0d4fadd68369610e.min.js",
		'python': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_python.dd3e2e25db7e7fb868d6.min.js",
		'q': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_q.4af8c1d9b07ea218977f.min.js",
		'r': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_r.783001720b360a8177a8.min.js",
		'rpm': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_rpm.79678546fb25c75e3208.min.js",
		'rst': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_rst.0fa19c56ae79c0b70c27.min.js",
		'ruby': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ruby.efce7fd348530776314b.min.js",
		'rust': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_rust.b148ea62a65dfe9f36c0.min.js",
		'sass': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sass.4e58ddf440975d3864f6.min.js",
		'scheme': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_scheme.52a48304089db7f7708e.min.js",
		'shell': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_shell.8dd47832ce011f080fb8.min.js",
		'sieve': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sieve.dc92cd9b919e5efb3c05.min.js",
		'slim': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_slim.ba0d300bced932d16420.min.js",
		'smalltalk': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_smalltalk.6dd6e1d419b04500b385.min.js",
		'smarty': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_smarty.428329a9fdb0d8be2a5f.min.js",
		'solr': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_solr.02f1fe78feb4a670b6cc.min.js",
		'soy': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_soy.8145a09e761fee4b0667.min.js",
		'sparql': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sparql.cf7a2190284c6ca0c11d.min.js",
		'spreadsheet': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_spreadsheet.eeeb35132617f7fa05e6.min.js",
		'sql': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sql.78a665f0a117e62e46f2.min.js",
		'stex': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_stex.777bff71a93e84be5096.min.js",
		'stylus': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_stylus.6ae0e46fb8c0750c644c.min.js",
		'swift': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_swift.2254c736e8a7f4f51e92.min.js",
		'tcl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tcl.13833d90818d7aa20cc6.min.js",
		'textile': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_textile.aa7de5d427d0474f6e14.min.js",
		'tiddlywiki': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tiddlywiki.efa88c874dc2653bb47e.min.js",
		'tiki': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tiki.6a8e59872a533ec09dae.min.js",
		'toml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_toml.4e099e2ec0d834eb7c04.min.js",
		'tornado': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tornado.feede7e509e683f0998f.min.js",
		'troff': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_troff.d31a17f22f8c679cf3e5.min.js",
		'ttcn': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ttcn.1bf6637cf05b45897ccd.min.js",
		'ttcn:cfg': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ttcn-cfg.273e541df1ddc66215ca.min.js",
		'turtle': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_turtle.4cf803c3d74096bfb82a.min.js",
		'twig': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_twig.628d79da0aea153a66fe.min.js",
		'vb': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vb.828b80361395c4e24aaf.min.js",
		'vbscript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vbscript.e19473b2883a8f5e9270.min.js",
		'velocity': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_velocity.09039c2d8f038c5046b2.min.js",
		'verilog': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_verilog.f12abef9991c95696bf5.min.js",
		'vhdl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vhdl.6438b130790bb71537f5.min.js",
		'vue': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vue.b7dca682b49e1cb360cf.min.js",
		'xml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_xml.c067158d12d43b9b96f7.min.js",
		'xquery': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_xquery.340466967c2bdf65fa66.min.js",
		'yaml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_yaml.7f71c0f462159ba81033.min.js",
		'z80': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_z80.73d5eb24ebcdece24ced.min.js"
	}		},

		notification_sounds: [{"value":"b2.mp3","label":"Ding","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/b2.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/b2.ogg"},{"value":"animal_stick.mp3","label":"Boing","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/animal_stick.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/animal_stick.ogg"},{"value":"been_tree.mp3","label":"Drop","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/been_tree.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/been_tree.ogg"},{"value":"complete_quest_requirement.mp3","label":"Ta-da","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/complete_quest_requirement.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/complete_quest_requirement.ogg"},{"value":"confirm_delivery.mp3","label":"Plink","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/confirm_delivery.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/confirm_delivery.ogg"},{"value":"flitterbug.mp3","label":"Wow","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/flitterbug.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/flitterbug.ogg"},{"value":"here_you_go_lighter.mp3","label":"Here you go","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/here_you_go_lighter.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/here_you_go_lighter.ogg"},{"value":"hi_flowers_hit.mp3","label":"Hi","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/hi_flowers_hit.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/hi_flowers_hit.ogg"},{"value":"knock_brush.mp3","label":"Knock Brush","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/knock_brush.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/knock_brush.ogg"},{"value":"save_and_checkout.mp3","label":"Whoa!","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/save_and_checkout.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/save_and_checkout.ogg"},{"value":"item_pickup.mp3","label":"Yoink","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/item_pickup.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/item_pickup.ogg"},{"value":"hummus.mp3","label":"Hummus","url":"https:\/\/slack.global.ssl.fastly.net\/7fa9\/sounds\/push\/hummus.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/hummus.ogg"},{"value":"none","label":"None"}],
		alert_sounds: [{"value":"frog.mp3","label":"Frog","url":"https:\/\/slack.global.ssl.fastly.net\/a34a\/sounds\/frog.mp3"}],
		call_sounds: [{"value":"call\/alert_v2.mp3","label":"Alert","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/alert_v2.mp3"},{"value":"call\/incoming_ring_v2.mp3","label":"Incoming ring","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/incoming_ring_v2.mp3"},{"value":"call\/outgoing_ring_v2.mp3","label":"Outgoing ring","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/outgoing_ring_v2.mp3"},{"value":"call\/pop_v2.mp3","label":"Incoming reaction","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/pop_v2.mp3"},{"value":"call\/they_left_call_v2.mp3","label":"They left call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/they_left_call_v2.mp3"},{"value":"call\/you_left_call_v2.mp3","label":"You left call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/you_left_call_v2.mp3"},{"value":"call\/they_joined_call_v2.mp3","label":"They joined call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/they_joined_call_v2.mp3"},{"value":"call\/you_joined_call_v2.mp3","label":"You joined call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/you_joined_call_v2.mp3"},{"value":"call\/confirmation_v2.mp3","label":"Confirmation","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/confirmation_v2.mp3"}],
		call_sounds_version: "v2",
				default_tz: "America\/Los_Angeles",
		
		feature_tinyspeck: false,
		feature_create_team_google_auth: false,
		feature_enterprise_custom_tos: false,
		feature_webapp_always_collect_initial_time_period_stats: false,
		feature_search_skip_context: false,
		feature_redux_batched_subscribe: false,
		feature_flannel_use_canary_sometimes: false,
		feature_store_members_in_redux: true,
		feature_store_files_in_redux: true,
		feature_cross_team_deeplink: true,
		feature_deprecate_window_cert: true,
		feature_deprecate_window_cert_block: true,
		feature_jumper_prevent_blur: true,
		feature_deprecate_files: false,
		feature_react_files: false,
		feature_file_threads: false,
		feature_threads_perf: false,
		feature_threads_api_stop_fe: true,
		feature_message_replies_inline: false,
		feature_subteam_members_diff: false,
		feature_a11y_keyboard_shortcuts: false,
		feature_email_ingestion: false,
		feature_msg_consistency: false,
		feature_jumper_sidebar: false,
		feature_sidebar_settings_button: false,
		feature_sidebar_context_menu: false,
		feature_attachments_inline: false,
		feature_fix_files: true,
		feature_channel_eventlog_client: true,
		feature_paging_api: false,
		feature_enterprise_app_teams: false,
		feature_enterprise_frecency: false,
		feature_ent_app_management_dashboard: false,
		feature_roles_from_roles_table: true,
		feature_entitlements: true,
		feature_precompute_org_user_counts: true,
		feature_payments_update_temp_user_activity: true,
		feature_grid_archive_link_fixes: true,
		feature_checkout_copy_edits: false,
		feature_stripe_elements_v3: false,
		feature_dunning: true,
		feature_invoice_dunning: true,
		feature_invoice_modification: true,
		feature_row_billing_direct_payments: false,
		feature_safeguard_org_retention: true,
		feature_ssi_checkout: true,
		feature_billing_edits: false,
		feature_dashboard_sortable_lists: false,
		feature_refactor_admin_stats: false,
		feature_guest_invitation_restrictions: true,
		feature_invite_only_workspaces: true,
		feature_mvch_conflict_popover_update: true,
		feature_leave_workspace_improvements: false,
		feature_enterprise_signup_name_tagging: true,
		feature_enterprise_org_wide_channels_section: false,
		feature_show_billing_active_for_grid: false,
		feature_find_your_workspace: false,
		feature_analytics_tooltip_copy: false,
		feature_sk_lato_cleanup: false,
		feature_saml_authn_key_expiry_date: true,
		feature_file_links_betterer: false,
		feature_session_duration_saved_message: false,
		feature_sso_jit_disabling: true,
		feature_channel_alert_words: false,
		feature_connecting_shared_channels_enterprise: false,
		feature_shared_channels_enterprise: false,
		feature_conversations_create_child: true,
		feature_snapshots_the_revenge: false,
		feature_mpim_channels: false,
		feature_esc_cancel_invitations_button: false,
		feature_esc_connecting_private_shared_channels: false,
		feature_conversations_create: true,
		feature_conversations_list: true,
		feature_esc_fix_dm_browser: false,
		feature_fix_displayname_guidelines: false,
		feature_newxpcreate_translations: true,
		feature_winssb_beta_channel: false,
		feature_slack_astronaut_i18n_jpn: true,
		feature_i18n_customer_stories: false,
		feature_cust_acq_i18n_tweaks: false,
		feature_presence_sub: true,
		feature_whitelist_zendesk_chat_widget: false,
		feature_live_support_free_plan: false,
		feature_slackbot_goes_to_college: false,
		feature_newxp_enqueue_message: true,
		feature_focus_mode: false,
		feature_star_shortcut: false,
		feature_show_jumper_scores: true,
		feature_force_ls_compression: false,
		feature_ignore_code_mentions: true,
		feature_name_tagging_client: true,
		feature_name_tagging_client_autocomplete: false,
		feature_name_tagging_client_topic_purpose: false,
		feature_use_imgproxy_resizing: true,
		feature_localization: true,
		feature_locale_ja_JP: true,
		feature_pseudo_locale: false,
		feature_share_mention_comment_cleanup: false,
		feature_external_files: false,
		feature_min_web: true,
		feature_electron_memory_logging: false,
		feature_tokenize_example_com: false,
		feature_zero_width_space_word_joiner: false,
		feature_channel_exports: false,
		feature_free_inactive_domains: true,
		feature_measure_css_usage: false,
		feature_take_profile_photo: false,
		feature_arugula: false,
		feature_texty_rewrite_on_paste: false,
		feature_deprecate_screenhero: true,
		feature_calls_esc_ui: true,
		feature_parsed_mrkdwn: false,
		feature_toggle_id_translation: true,
		feature_id_translation_copy_updates: true,
		feature_emoji_menu_tuning: false,
		feature_default_shared_channels: false,
		feature_react_lfs: false,
		feature_log_quickswitcher_queries: false,
		feature_enable_mdm: true,
		feature_message_menus: true,
		feature_sli_recaps: true,
		feature_sli_recaps_interface: true,
		feature_new_message_click_logging: true,
		feature_announce_only_channels: false,
		feature_app_permissions_backend_enterprise: false,
		feature_token_ip_whitelist: true,
		feature_hide_join_leave: false,
		feature_rollback_to_mapping: false,
		feature_update_emoji_to_v4: false,
		feature_allow_intra_word_formatting: true,
		feature_allow_cjk_autocomplete: true,
		feature_allow_split_word: false,
		feature_slim_scrollbar: false,
		feature_sli_briefing: true,
		feature_sli_channel_insights: true,
		feature_sli_file_search: true,
		feature_react_search: false,
		feature_sli_home: false,
		feature_sidebar_filters: false,
		feature_react_messages: false,
		feature_react_member_profile_card: false,
		feature_store_membership_in_redux: false,
		feature_hide_membership_counts_in_channel_browser: false,
		feature_mpdm_default_fe: false,
		feature_channel_notif_dialog_update: false,
		feature_delay_thread_mark: true,
		feature_api_admin_page: true,
		feature_api_admin_page_not_ent: false,
		feature_oauth_react_wta: false,
		feature_app_index: false,
		feature_untangle_app_directory_templates: true,
		feature_app_profile_app_site_link: false,
		feature_custom_app_installs: false,
		feature_gdrive_do_not_install_by_default: true,
		feature_delete_moved_channels: false,
		feature_solr_discoverability: false,
		feature_ms_msg_handlers_profiling: true,
		feature_cross_workspace_quick_switcher_prototype: false,
		feature_org_quick_switcher: false,
		feature_ms_latest: true,
		feature_no_preload_video: true,
		feature_react_emoji_picker_frecency: false,
		feature_app_space: true,
		feature_app_space_permissions_tab: false,
		feature_app_canvases: false,
		feature_queue_metrics: false,
		feature_trace_reason: false,
		feature_stop_loud_channel_mentions: false,
		feature_message_input_byte_limit: false,
		feature_perfectrics: false,
		feature_automated_perfectrics: false,
		feature_link_buttons: true,
		feature_nudge_team_creators: false,
		feature_onedrive_picker: false,
		feature_lesson_onboarding: true,
		feature_skip_onboarding_task_i18n: false,
		feature_lazy_grid_teams_menu: true,
		feature_less_light_up: true,
		feature_ent_user_teams_validate: true,
		feature_less_c_ids_fetch: true,
		feature_less_history_when_muted: false,
		feature_less_history_when_changed: false,
		feature_delete_team_and_apps: true,
		feature_email_forwarding: true,
		feature_opt_out_react_messages_pref: false,
		feature_pjpeg: false,
		feature_pdf_thumb: true,
		feature_async_uploads_jq: false,
		feature_apps_manage_permissions_scope_changes: true,
		feature_app_dir_only_default_true: false,
		feature_reminder_cross_workspace: false,
		feature_speedy_boot_handlebars: false,
		feature_unified_app_display: false,
		feature_wta_management_modal: false,
		feature_channel_invite_modal_cannot_join: false,
		feature_cancel_survey: true,
		feature_promo_code_sys: true,
		feature_expiring_credits: false,
		feature_sonic: false,
		feature_modern_rtm_dispatch: true,
		feature_flannel_channels_v0: false,
		feature_shortcuts_flexpane: true,
		feature_app_directory_home_page_redesign: true,
		feature_hidden_wksp_unfurls: false,
		feature_guest_wksp_unfurls: false,
		feature_charging_vat: false,
		feature_workspace_scim_management: false,
		feature_billing_ff_for_invoice_pdfs: false,
		feature_channel_updated_event: false,
		feature_email_previewer: false,
		feature_turn_mpdm_notifs_on: false,
		feature_browser_dragndrop: false,
		feature_granular_shared_channel_perms: false,
		feature_notification_method: true,
		feature_org_detailed_thread_mentions: true,
		feature_force_production_channel: false,
		feature_quill_upgrade: false,
		feature_inline_emoji: false,
		feature_agnostic_autoslugging: true,
		feature_slug_tooltips: true,
		feature_increase_msg_input_height: false,
		feature_shortcuts_prompt: true,
		feature_accessible_dialogs: true,
		feature_pending_channel_string: false,
		feature_app_actions: false,
		feature_app_actions_fe: false,
		feature_shared_channel_free_trial_flow: false,
		feature_i18n_compliance_links: false,
		feature_calls_clipboard_broadcasting_optin: false,
		feature_unified_autocomplete: false,
		feature_screen_share_needs_aero: false,
		feature_i18n_calls_upgrading: false,
		feature_calls_disable_iss_admin: false,
		feature_msg_lim_banner_refactor: false,

	client_logs: {"0":{"numbers":[0],"user_facing":false},"@scott":{"numbers":[2,4,37,58,67,141,142,389,481,488,529,667,773,808,888,999,1701],"owner":"@scott"},"@eric":{"numbers":[2,23,47,48,72,73,82,91,93,96,365,438,552,777,794],"owner":"@eric"},"2":{"owner":"@scott \/ @eric","numbers":[2],"user_facing":false},"4":{"owner":"@scott","numbers":[4],"user_facing":false},"5":{"channels":"#dhtml","numbers":[5],"user_facing":false},"23":{"owner":"@eric","numbers":[23],"user_facing":false},"sounds":{"owner":"@scott","name":"sounds","numbers":[37]},"37":{"owner":"@scott","name":"sounds","numbers":[37],"user_facing":true},"47":{"owner":"@eric","numbers":[47],"user_facing":false},"48":{"owner":"@eric","numbers":[48],"user_facing":false},"Message History":{"owner":"@scott","name":"Message History","numbers":[58]},"58":{"owner":"@scott","name":"Message History","numbers":[58],"user_facing":true},"67":{"owner":"@scott","numbers":[67],"user_facing":false},"72":{"owner":"@eric","numbers":[72],"user_facing":false},"73":{"owner":"@eric","numbers":[73],"user_facing":false},"82":{"owner":"@eric","numbers":[82],"user_facing":false},"@shinypb":{"owner":"@shinypb","numbers":[88,1000,1989,1990,1991,1996]},"88":{"owner":"@shinypb","numbers":[88],"user_facing":false},"91":{"owner":"@eric","numbers":[91],"user_facing":false},"93":{"owner":"@eric","numbers":[93],"user_facing":false},"96":{"owner":"@eric","numbers":[96],"user_facing":false},"@steveb":{"owner":"@steveb","numbers":[99]},"99":{"owner":"@steveb","numbers":[99],"user_facing":false},"Channel Marking (MS)":{"owner":"@scott","name":"Channel Marking (MS)","numbers":[141]},"141":{"owner":"@scott","name":"Channel Marking (MS)","numbers":[141],"user_facing":true},"Channel Marking (Client)":{"owner":"@scott","name":"Channel Marking (Client)","numbers":[142]},"142":{"owner":"@scott","name":"Channel Marking (Client)","numbers":[142],"user_facing":true},"365":{"owner":"@eric","numbers":[365],"user_facing":false},"389":{"owner":"@scott","numbers":[389],"user_facing":false},"438":{"owner":"@eric","numbers":[438],"user_facing":false},"@rowan":{"numbers":[444,666],"owner":"@rowan"},"444":{"owner":"@rowan","numbers":[444],"user_facing":false},"481":{"owner":"@scott","numbers":[481],"user_facing":false},"488":{"owner":"@scott","numbers":[488],"user_facing":false},"529":{"owner":"@scott","numbers":[529],"user_facing":false},"552":{"owner":"@eric","numbers":[552],"user_facing":false},"dashboard":{"owner":"@ac \/ @bruce \/ @kylestetz \/ @nic \/ @rowan","channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666]},"@ac":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@ac"},"@bruce":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@bruce"},"@kylestetz":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@kylestetz"},"@nic":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@nic"},"666":{"owner":"@ac \/ @bruce \/ @kylestetz \/ @nic \/ @rowan","channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"user_facing":false},"667":{"owner":"@scott","numbers":[667],"user_facing":false},"773":{"owner":"@scott","numbers":[773],"user_facing":false},"777":{"owner":"@eric","numbers":[777],"user_facing":false},"794":{"owner":"@eric","numbers":[794],"user_facing":false},"Client Responsiveness":{"owner":"@scott","name":"Client Responsiveness","user_facing":false,"numbers":[808]},"808":{"owner":"@scott","name":"Client Responsiveness","user_facing":false,"numbers":[808]},"Message Pane Scrolling":{"owner":"@scott","name":"Message Pane Scrolling","numbers":[888]},"888":{"owner":"@scott","name":"Message Pane Scrolling","numbers":[888],"user_facing":true},"Unread banner and divider":{"owner":"@scott","name":"Unread banner and divider","numbers":[999]},"999":{"owner":"@scott","name":"Unread banner and divider","numbers":[999],"user_facing":true},"1000":{"owner":"@shinypb","numbers":[1000],"user_facing":false},"Duplicate badges (desktop app icons)":{"owner":"@scott","name":"Duplicate badges (desktop app icons)","numbers":[1701]},"1701":{"owner":"@scott","name":"Duplicate badges (desktop app icons)","numbers":[1701],"user_facing":true},"Members":{"owner":"@fearon","name":"Members","numbers":[1975]},"@fearon":{"owner":"@fearon","name":"Members","numbers":[1975,98765]},"1975":{"owner":"@fearon","name":"Members","numbers":[1975],"user_facing":true},"lazy loading":{"owner":"@shinypb","channels":"#devel-flannel","name":"lazy loading","numbers":[1989]},"1989":{"owner":"@shinypb","channels":"#devel-flannel","name":"lazy loading","numbers":[1989],"user_facing":true},"thin_channel_membership":{"owner":"@shinypb","channels":"#devel-infrastructure","name":"thin_channel_membership","numbers":[1990]},"1990":{"owner":"@shinypb","channels":"#devel-infrastructure","name":"thin_channel_membership","numbers":[1990],"user_facing":true},"stats":{"owner":"@shinypb","channels":"#team-infrastructure","name":"stats","numbers":[1991]},"1991":{"owner":"@shinypb","channels":"#team-infrastructure","name":"stats","numbers":[1991],"user_facing":true},"ms":{"owner":"@shinypb","name":"ms","numbers":[1996]},"1996":{"owner":"@shinypb","name":"ms","numbers":[1996],"user_facing":true},"shared_channels_connection":{"owner":"@jim","name":"shared_channels_connection","numbers":[1999]},"@jim":{"owner":"@jim","name":"shared_channels_connection","numbers":[1999]},"1999":{"owner":"@jim","name":"shared_channels_connection","numbers":[1999],"user_facing":false},"dnd":{"owner":"@patrick","channels":"dhtml","name":"dnd","numbers":[2002]},"@patrick":{"owner":"@patrick","channels":"dhtml","name":"dnd","numbers":[2002,2003,2004,2005,2006,2007]},"2002":{"owner":"@patrick","channels":"dhtml","name":"dnd","numbers":[2002],"user_facing":true},"2003":{"owner":"@patrick","channels":"dhtml","numbers":[2003],"user_facing":false},"Threads":{"owner":"@patrick","channels":"#feat-threads, #devel-threads","name":"Threads","numbers":[2004]},"2004":{"owner":"@patrick","channels":"#feat-threads, #devel-threads","name":"Threads","numbers":[2004],"user_facing":true},"2005":{"owner":"@patrick","numbers":[2005],"user_facing":false},"Reactions":{"owner":"@patrick","name":"Reactions","numbers":[2006]},"2006":{"owner":"@patrick","name":"Reactions","numbers":[2006],"user_facing":true},"TSSSB.focusTabAndSwitchToChannel":{"owner":"@patrick","name":"TSSSB.focusTabAndSwitchToChannel","numbers":[2007]},"2007":{"owner":"@patrick","name":"TSSSB.focusTabAndSwitchToChannel","numbers":[2007],"user_facing":false},"Presence Detection":{"owner":"@ainjii","name":"Presence Detection","numbers":[2017]},"@ainjii":{"owner":"@ainjii","name":"Presence Detection","numbers":[2017,8675309]},"2017":{"owner":"@ainjii","name":"Presence Detection","numbers":[2017],"user_facing":true},"mc_sibs":{"name":"mc_sibs","numbers":[9999]},"9999":{"name":"mc_sibs","numbers":[9999],"user_facing":false},"98765":{"owner":"@fearon","numbers":[98765],"user_facing":false},"8675309":{"owner":"@ainjii","numbers":[8675309],"user_facing":false},"@monty":{"owner":"@monty","numbers":[6532]},"6532":{"owner":"@monty","numbers":[6532],"user_facing":false}},


		img: {
			app_icon: 'https://a.slack-edge.com/272a/img/slack_growl_icon.png'
		},
		page_needs_custom_emoji: false,
		page_needs_enterprise: false	};

	
	
	
	
	
	// i18n locale map (eg: maps Slack `ja-jp` to ZD `ja`)
			boot_data.slack_to_zd_locale = {"en-us":"en-us","fr-fr":"fr-fr","de-de":"de","es-es":"es","ja-jp":"ja"};
	
	// client boot data
		
					boot_data.should_use_flannel = true;
		boot_data.ms_connect_url = "wss:\/\/mpmulti-rBjg.lb.slack-msgs.com\/?flannel=2&token=xoxs-238068740516-239494185927-257070208578-c66b928e42";
				boot_data.page_has_incomplete_user_model = true;
		boot_data.flannel_server_pool = "random";
		
	
	
	
	
	
				
	
//-->
</script>
	








	<!-- output_js "libs" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/emoji.f19f28988996a742b130.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-core_required_libs.434900278509ee5dc46d.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

	<!-- output_js "application" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/modern.vendor.aa41e8febd80fa22513e.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/application.a96b8dc70212462dc26a.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

	<!-- output_js "core" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-core_required_ts.4fafab590fecdda3536f.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/TS.web.5056d4e1c7a291374dee.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

	<!-- output_js "core_web" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-core_web.c76020a8930b81cb8c9e.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

	<!-- output_js "secondary" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-secondary_a_required.af14c5eac2eaa217cf91.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-secondary_b_required.eb136ff32a72700990eb.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

			
	
	<!-- output_js "regular" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/TS.web.comments.83e35efdab6c80c17a4e.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/TS.web.file.b49727c8257c81f9ed17.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/codemirror.min.41c3faeb73621d67a666.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/simple.45192890ef119b00f332.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/codemirror_load.0e1999f4ce5168fec6cb.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>


		<script type="text/javascript">
				TS.clog.setTeam('T7020MSF6');
				TS.clog.setUser('U71EJ5FT9');
			</script>

			<script type="text/javascript">
	<!--
		boot_data.page_needs_custom_emoji = true;

		boot_data.file = {"id":"F88JBS1KR","created":1512013619,"timestamp":1512013619,"name":"lab11finished.py","title":"lab11finished.py","mimetype":"text\/plain","filetype":"python","pretty_type":"Python","user":"U71477WFQ","editable":true,"size":24026,"mode":"snippet","is_external":false,"external_type":"","is_public":true,"public_url_shared":false,"display_as_bot":false,"username":"","url_private":"https:\/\/files.slack.com\/files-pri\/T7020MSF6-F88JBS1KR\/lab11finished.py","url_private_download":"https:\/\/files.slack.com\/files-pri\/T7020MSF6-F88JBS1KR\/download\/lab11finished.py","permalink":"https:\/\/byu-dl-f17.slack.com\/files\/U71477WFQ\/F88JBS1KR\/lab11finished.py","permalink_public":"https:\/\/slack-files.com\/T7020MSF6-F88JBS1KR-2578eac0a9","edit_link":"https:\/\/byu-dl-f17.slack.com\/files\/U71477WFQ\/F88JBS1KR\/lab11finished.py\/edit","preview":"#!\/usr\/bin\/python\n# -*- coding: utf-8\n\nimport torch\nfrom torch.autograd import Variable","preview_highlight":"\u003Cdiv class=\"CodeMirror cm-s-default CodeMirrorServer\" oncopy=\"if(event.clipboardData){event.clipboardData.setData('text\/plain',window.getSelection().toString().replace(\/\\u200b\/g,''));event.preventDefault();event.stopPropagation();}\"\u003E\n\u003Cdiv class=\"CodeMirror-code\"\u003E\n\u003Cdiv\u003E\u003Cpre\u003E\u003Cspan class=\"cm-comment\"\u003E#!\/usr\/bin\/python\u003C\/span\u003E\u003C\/pre\u003E\u003C\/div\u003E\n\u003Cdiv\u003E\u003Cpre\u003E\u003Cspan class=\"cm-comment\"\u003E# -*- coding: utf-8\u003C\/span\u003E\u003C\/pre\u003E\u003C\/div\u003E\n\u003Cdiv\u003E\u003Cpre\u003E&#8203;\u003C\/pre\u003E\u003C\/div\u003E\n\u003Cdiv\u003E\u003Cpre\u003E\u003Cspan class=\"cm-keyword\"\u003Eimport\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003Etorch\u003C\/span\u003E\u003C\/pre\u003E\u003C\/div\u003E\n\u003Cdiv\u003E\u003Cpre\u003E\u003Cspan class=\"cm-keyword\"\u003Efrom\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003Etorch\u003C\/span\u003E.\u003Cspan class=\"cm-property\"\u003Eautograd\u003C\/span\u003E \u003Cspan class=\"cm-keyword\"\u003Eimport\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003EVariable\u003C\/span\u003E\u003C\/pre\u003E\u003C\/div\u003E\n\u003C\/div\u003E\n\u003C\/div\u003E\n","lines":548,"lines_more":543,"preview_is_truncated":true,"channels":["C7TR9PFNH"],"groups":[],"ims":[],"comments_count":1,"initial_comment":{"id":"Fc87HEJ8NQ","created":1512013619,"timestamp":1512013619,"user":"U71477WFQ","is_intro":true,"comment":"\u003C!channel\u003E Here's the new scaffolding file. It will be up on \u003Chttp:\/\/liftothers.org|liftothers.org\u003E by tomorrow. I've got implementations for doing it in LSTM and GRU. If you stil have time I would recommend implementing the Attention Mechanism as found in \u003Chttp:\/\/pytorch.org|pytorch.org\u003E's NMT tutorial. That will reduce the amount of time and number of parameters needed. Best!"}};
		boot_data.file.comments = [{"id":"Fc87HEJ8NQ","created":1512013619,"timestamp":1512013619,"user":"U71477WFQ","is_intro":true,"comment":"\u003C!channel\u003E Here's the new scaffolding file. It will be up on \u003Chttp:\/\/liftothers.org|liftothers.org\u003E by tomorrow. I've got implementations for doing it in LSTM and GRU. If you stil have time I would recommend implementing the Attention Mechanism as found in \u003Chttp:\/\/pytorch.org|pytorch.org\u003E's NMT tutorial. That will reduce the amount of time and number of parameters needed. Best!"}];

		

		var g_editor;

		var code_wrap_long_lines = true;

		$(function(){

			var wrap_long_lines = !!code_wrap_long_lines;

			g_editor = CodeMirror(function(elt){
				var content = document.getElementById("file_contents");
				content.parentNode.replaceChild(elt, content);
			}, {
				value: $('#file_contents').text(),
				lineNumbers: true,
				matchBrackets: true,
				indentUnit: 4,
				indentWithTabs: true,
				enterMode: "keep",
				tabMode: "shift",
				viewportMargin: 10,
				readOnly: true,
				lineWrapping: wrap_long_lines
			});

			// past a certain point, CodeMirror rendering may simply stop working.
			// start having CodeMirror use its Long List View-style scolling at >= max_lines.
			var max_lines = 8192;

			var snippet_lines;

			// determine # of lines, limit height accordingly
			if (g_editor.doc && g_editor.doc.lineCount) {
				snippet_lines = parseInt(g_editor.doc.lineCount());
				var new_height;
				if (snippet_lines) {
					// we want the CodeMirror container to collapse around short snippets.
					// however, we want to let it auto-expand only up to a limit, at which point
					// we specify a fixed height so CM can display huge snippets without dying.
					// this is because CodeMirror works nicely with no height set, but doesn't
					// scale (big file performance issue), and doesn't work with min/max-height -
					// so at some point, we have to set an absolute height to kick it into its
					// smart / partial "Long List View"-style rendering mode.
					if (snippet_lines < max_lines) {
						new_height = 'auto';
					} else {
						new_height = (max_lines * 0.875) + 'rem'; // line-to-rem ratio
					}
					var line_count = Math.min(snippet_lines, max_lines);
					TS.info('Applying height of ' + new_height + ' to container for this snippet of ' + snippet_lines + (snippet_lines === 1 ? ' line' : ' lines'));
					$('#page .CodeMirror').height(new_height);
				}
			}

			$('#file_preview_wrap_cb').bind('change', function(e) {
				code_wrap_long_lines = $(this).prop('checked');
				g_editor.setOption('lineWrapping', code_wrap_long_lines);
			})

			$('#file_preview_wrap_cb').prop('checked', wrap_long_lines);

			CodeMirror.switchSlackMode(g_editor, "python");
		});

		
		$('#file_comment').css('overflow', 'hidden').autogrow();
	//-->
	</script>


	<script type="text/javascript">TS.boot(boot_data);</script>

<style>.color_9f69e7:not(.nuc) {color:#9F69E7;}.color_4bbe2e:not(.nuc) {color:#4BBE2E;}.color_e7392d:not(.nuc) {color:#E7392D;}.color_3c989f:not(.nuc) {color:#3C989F;}.color_674b1b:not(.nuc) {color:#674B1B;}.color_e96699:not(.nuc) {color:#E96699;}.color_e0a729:not(.nuc) {color:#E0A729;}.color_684b6c:not(.nuc) {color:#684B6C;}.color_5b89d5:not(.nuc) {color:#5B89D5;}.color_2b6836:not(.nuc) {color:#2B6836;}.color_99a949:not(.nuc) {color:#99A949;}.color_df3dc0:not(.nuc) {color:#DF3DC0;}.color_4cc091:not(.nuc) {color:#4CC091;}.color_9b3b45:not(.nuc) {color:#9B3B45;}.color_d58247:not(.nuc) {color:#D58247;}.color_bb86b7:not(.nuc) {color:#BB86B7;}.color_5a4592:not(.nuc) {color:#5A4592;}.color_db3150:not(.nuc) {color:#DB3150;}.color_235e5b:not(.nuc) {color:#235E5B;}.color_9e3997:not(.nuc) {color:#9E3997;}.color_53b759:not(.nuc) {color:#53B759;}.color_c386df:not(.nuc) {color:#C386DF;}.color_385a86:not(.nuc) {color:#385A86;}.color_a63024:not(.nuc) {color:#A63024;}.color_5870dd:not(.nuc) {color:#5870DD;}.color_ea2977:not(.nuc) {color:#EA2977;}.color_50a0cf:not(.nuc) {color:#50A0CF;}.color_d55aef:not(.nuc) {color:#D55AEF;}.color_d1707d:not(.nuc) {color:#D1707D;}.color_43761b:not(.nuc) {color:#43761B;}.color_e06b56:not(.nuc) {color:#E06B56;}.color_8f4a2b:not(.nuc) {color:#8F4A2B;}.color_902d59:not(.nuc) {color:#902D59;}.color_de5f24:not(.nuc) {color:#DE5F24;}.color_a2a5dc:not(.nuc) {color:#A2A5DC;}.color_827327:not(.nuc) {color:#827327;}.color_3c8c69:not(.nuc) {color:#3C8C69;}.color_8d4b84:not(.nuc) {color:#8D4B84;}.color_84b22f:not(.nuc) {color:#84B22F;}.color_4ec0d6:not(.nuc) {color:#4EC0D6;}.color_e23f99:not(.nuc) {color:#E23F99;}.color_e475df:not(.nuc) {color:#E475DF;}.color_619a4f:not(.nuc) {color:#619A4F;}.color_a72f79:not(.nuc) {color:#A72F79;}.color_7d414c:not(.nuc) {color:#7D414C;}.color_aba727:not(.nuc) {color:#ABA727;}.color_965d1b:not(.nuc) {color:#965D1B;}.color_4d5e26:not(.nuc) {color:#4D5E26;}.color_dd8527:not(.nuc) {color:#DD8527;}.color_bd9336:not(.nuc) {color:#BD9336;}.color_e85d72:not(.nuc) {color:#E85D72;}.color_dc7dbb:not(.nuc) {color:#DC7DBB;}.color_bc3663:not(.nuc) {color:#BC3663;}.color_9d8eee:not(.nuc) {color:#9D8EEE;}.color_8469bc:not(.nuc) {color:#8469BC;}.color_73769d:not(.nuc) {color:#73769D;}.color_b14cbc:not(.nuc) {color:#B14CBC;}</style>

<!-- slack-www-hhvm-04457c75b0a3d59e9 / 2017-11-30 11:44:34 / v3b12e0970c59c74eaf054e23d960f9c4a0dcd138 / B:H -->


</body>
</html>